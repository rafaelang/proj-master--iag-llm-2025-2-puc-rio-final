"""P5 · Consolidar golden v0.6 (perguntas_v06.json) = 44 congeladas v0.5b + candidatos P5.

Preserva as 44 perguntas da v0.5b TEXTUALMENTE (IDs 1-44, docs_esperados intactos)
e adiciona os candidatos P5 (IDs 45+). A resposta_esperada e a fonte ficam em
CAMPO SEPARADO (nao poluem o schema do golden; servem de auditoria/revisao humana
antes de congelar). Valida:
  - docs_esperados existem no corpus (manifesto);
  - sem duplicata de pergunta (normalizada) entre congeladas e novas;
  - estrato e deve_abster coerentes (adversarial => docs vazio, deve_abster true).

Saida: data/golden_set/rag/perguntas_v06.json (golden expandido, congelavel)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from projeto_final import config
from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.retrieve import recuperar

CAND = config.PROCESSED_DIR / "v05b_ab" / "golden_v06_candidatos.json"
BASE = config.GOLDEN_SET_DIR / "rag" / "perguntas_v05b.json"
SAIDA = config.GOLDEN_SET_DIR / "rag" / "perguntas_v06.json"


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]", "", q.lower())


def _alcancavel(pergunta: str, docs: list[str], chunks: list[dict]) -> bool:
    """P5: doc esperado precisa estar no pool (top-60) — senão a pergunta tem
    formulação genérica (ex.: 'o que o exercício 4 sugere') e criaria um caso
    injusto como as 5Q residuais. Irrecuperável => descarta o candidato."""
    if not docs:
        return True  # adversarial (fora do corpus)
    rec = recuperar(pergunta, chunks, top_k=60, base=None)
    return any(c["doc_id"] in docs for c in rec)


def main() -> None:
    if not CAND.exists():
        raise SystemExit(f"sem candidatos: {CAND}")
    candidatos = json.loads(CAND.read_text(encoding="utf-8"))
    base = json.loads(BASE.read_text(encoding="utf-8"))
    base_perg = base["perguntas"]
    chunks = carregar_chunks()

    # chave: pergunta normalizada das congeladas
    norms = {norm(q["pergunta"]): q for q in base_perg}
    novos = []
    descartadas = []
    for c in candidatos:
        k = norm(c["pergunta"])
        if k in norms:
            descartadas.append(c["pergunta"][:70])
            continue
        if not _alcancavel(c["pergunta"], c["docs_esperados"], chunks):
            descartadas.append(f"[irrecuperavel] {c['pergunta'][:70]}")
            continue
        novos.append({
            "id": None,  # preenchido abaixo
            "pergunta": c["pergunta"],
            "estrato": c["estrato"],
            "docs_esperados": sorted(c["docs_esperados"]),
            "deve_abster": bool(c["deve_abster"]),
            "_auditoria": {  # metadado de revisao, nao faz parte do golden avaliado
                "resposta_esperada": c.get("resposta_esperada", ""),
                "fonte": c.get("fonte_chunks", [])[0] if c.get("fonte_chunks") else "",
                "motivo": c.get("motivo", ""),
            },
        })

    # distribui IDs sequenciais apos as 44 congeladas
    prox = max(int(q["id"]) for q in base_perg) + 1
    for q in novos:
        q["id"] = prox
        prox += 1

    final = base_perg + novos

    # ---- validações (mesmas do test_v05b) ----
    manifest = json.loads(
        (config.GOLDEN_SET_DIR / "corpus_manifest.json").read_text(encoding="utf-8"))
    corpus = {e["arquivo"] for e in manifest["corpus"]}
    ausentes = {d for q in final for d in q["docs_esperados"] if d not in corpus}
    assert not ausentes, f"docs_esperados fora do corpus: {ausentes}"

    # congeladas preservadas textualmente
    orig = {q["id"]: q for q in base_perg}
    for q in final:
        if q["id"] in orig:
            assert q["pergunta"] == orig[q["id"]]["pergunta"]
            assert q["docs_esperados"] == orig[q["id"]]["docs_esperados"]

    # adversarial => docs vazio + deve_abster
    for q in final:
        if q["estrato"] == "adversarial":
            assert not q["docs_esperados"] and q["deve_abster"], f"adversarial invalido #{q['id']}"

    # sem duplicata
    seen = {}
    for q in final:
        k = norm(q["pergunta"])
        assert k not in seen, f"duplicata: {q['pergunta'][:60]}"
        seen[k] = q["id"]

    from collections import Counter
    estratos = Counter(q["estrato"] for q in final)
    meta = {
        "dominio": "materiais do Master IAG & LLM (PUC-Rio)",
        "estratos": ["rotineira", "composta", "negativa", "adversarial"],
        "metricas": "recall@k (docs_esperados recuperados) + taxa de abstenção correta (absteve == deve_abster) + juiz de correção",
        "origem": "v0.6 - P5: golden expandido (44 congeladas v0.5b + novos ancorados no corpus)",
        "nota": f"IDs 1-44 congeladas da v0.5b (textualmente preservadas). IDs 45+ gerados no P5 "
                f"com curadoria de ancoragem (resposta_esperada em _auditoria, fora do schema "
                f"avaliado). Estratos: {dict(estratos)}. Critério de rótulo: "
                f"data/golden_set/adaptacao/criterio_rotulo.md (congelado).",
    }
    out = {"meta": meta, "perguntas": final}
    SAIDA.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"base={len(base_perg)} novos={len(novos)} descartadas={len(descartadas)} total={len(final)}")
    print(f"estratos: {dict(estratos)}")
    if descartadas:
        print("descartadas (dup com congeladas):", descartadas[:5])
    print(f"salvo em {SAIDA}")


if __name__ == "__main__":
    main()
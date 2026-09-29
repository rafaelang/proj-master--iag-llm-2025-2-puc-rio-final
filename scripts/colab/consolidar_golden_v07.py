"""v0.7 · Consolidar golden expandido (perguntas_v07.json) = 132 congeladas + candidatos P7.

Preserva as 132 perguntas da v0.6 TEXTUALMENTE (IDs 1-132, docs_esperados intactos)
e adiciona os candidatos P7 (IDs 133+). A resposta_esperada e a fonte ficam em
CAMPO SEPARADO (_auditoria). Valida (mesmas regras do test_v06):
  - docs_esperados existem no corpus (manifesto);
  - sem duplicata de pergunta (normalizada) entre congeladas e novas;
  - estrato/deve_abster coerentes (adversarial => docs vazio, deve_abster true);
  - novas com doc esperado ALCANÇÁVEL no pool (top-60) — descarta irrecuperáveis.

Saida: data/golden_set/rag/perguntas_v07.json (golden congelável — entra no Git)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from projeto_final import config
from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.retrieve import recuperar

CAND = config.PROCESSED_DIR / "v07_ab" / "golden_v07_candidatos.json"
BASE = config.GOLDEN_SET_DIR / "rag" / "perguntas_v06.json"
SAIDA = config.GOLDEN_SET_DIR / "rag" / "perguntas_v07.json"


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]", "", q.lower())


def _alcancavel(pergunta: str, docs: list[str], chunks: list[dict]) -> bool:
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
            "id": None,
            "pergunta": c["pergunta"],
            "estrato": c["estrato"],
            "docs_esperados": sorted(c["docs_esperados"]),
            "deve_abster": bool(c["deve_abster"]),
            "_auditoria": {
                "resposta_esperada": c.get("resposta_esperada", ""),
                "fonte": c.get("fonte_chunks", [])[0] if c.get("fonte_chunks") else "",
                "motivo": c.get("motivo", ""),
            },
        })

    prox = max(int(q["id"]) for q in base_perg) + 1
    for q in novos:
        q["id"] = prox
        prox += 1

    final = base_perg + novos

    # ---- validações ----
    manifest = json.loads(
        (config.GOLDEN_SET_DIR / "corpus_manifest.json").read_text(encoding="utf-8"))
    corpus = {e["arquivo"] for e in manifest["corpus"]}
    ausentes = {d for q in final for d in q["docs_esperados"] if d not in corpus}
    assert not ausentes, f"docs_esperados fora do corpus: {ausentes}"

    orig = {q["id"]: q for q in base_perg}
    for q in final:
        if q["id"] in orig:
            assert q["pergunta"] == orig[q["id"]]["pergunta"]
            assert q["docs_esperados"] == orig[q["id"]]["docs_esperados"]

    for q in final:
        if q["estrato"] == "adversarial":
            assert not q["docs_esperados"] and q["deve_abster"], f"adversarial invalido #{q['id']}"

    seen = {}
    for q in final:
        k = norm(q["pergunta"])
        assert k not in seen, f"duplicata: {q['pergunta'][:60]}"
        seen[k] = q["id"]

    estratos = Counter(q["estrato"] for q in final)
    meta = {
        "dominio": "materiais do Master IAG & LLM (PUC-Rio)",
        "estratos": ["rotineira", "composta", "negativa", "adversarial"],
        "metricas": "recall@k (docs_esperados recuperados) + taxa de abstenção correta (absteve == deve_abster) + juiz de correção",
        "origem": "v0.7 - receita P5: golden expandido (132 congeladas v0.6 + novos ancorados no corpus)",
        "nota": f"IDs 1-132 congeladas da v0.6 (textualmente preservadas). IDs 133+ gerados no P7 "
                f"com curadoria de ancoragem (resposta_esperada em _auditoria, fora do schema "
                f"avaliado). Estratos: {dict(estratos)}. Critério de rótulo: "
                f"data/golden_set/adaptacao/criterio_rotulo.md (congelado).",
    }
    out = {"meta": meta, "perguntas": final}
    SAIDA.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"base={len(base_perg)} novos={len(novos)} descartadas={len(descartadas)} total={len(final)}")
    print(f"estratos: {dict(estratos)}")
    if descartadas:
        print("descartadas:", descartadas[:8])
    print(f"salvo em {SAIDA}")


if __name__ == "__main__":
    main()
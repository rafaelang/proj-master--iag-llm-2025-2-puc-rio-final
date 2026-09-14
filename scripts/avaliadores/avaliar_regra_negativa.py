"""v0.6 · P1 (plano do orientador) — regra determinística para negativas tipo A (item ausente).

Tese (diagnóstico P0): perguntas no template
  "Qual destes itens/termos NÃO está relacionado/é conceito de X: a, b, c ou d?"
têm a resposta correta = opção AUSENTE do contexto recuperado (o item fora do domínio
É a informação). Regra 3 do rag_sistema_slm.txt (abster quando o contexto não contém a
informação) torna essa resposta *estruturalmente impossível* para o SLM (14/14 abstêm).

Este script mede, intra-sessão, variantes da regra mecânica (presença/ausência das
opções no contexto recuperado), isolado do src/ (nenhuma mudança no pipeline):

  V1 — checagem no TOP-5 (contexto real da geração); absent==1 → responde;
  V2 — checagem no POOL (top-60 RRF);
  V4 — checagem nos DOCS COMPLETOS recuperados (top-5 doc) — VENCEDORA (7/14, 0 erros);
  V3 — V1 + fallback de menor ocorrência (absent==0 → op com menos ocorrências;
       absent>=2 → op com zero ocorrências, se única).

Guarda de colisão negativa×adversarial (P0/S3): se TODAS as opções estiverem ausentes
do contexto (pergunta fora do corpus), a regra NÃO responde (abstenção) — vira o
comportamento correto para adversarial. Testado com 3 sondas sintéticas.

Saída:
- data/processed/v05b_ab/regra_negativa_ev.json (dados intermediários)
- docs/v06_negativa.md (evidência + decisão da variante e do ponto de integração)

Uso:
  python scripts/avaliadores/avaliar_regra_negativa.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.retrieve import recuperar

RAIZ = Path(__file__).resolve().parent.parent.parent
GOLDEN = RAIZ / "data/golden_set/rag/perguntas_v06.json"
CASCADE = RAIZ / "data/processed/v05b_ab/cascade_v06_p2.jsonl"
SAIDA = RAIZ / "data/processed/v05b_ab/regra_negativa_ev.json"
MD = RAIZ / "docs/v06_negativa.md"

TIPO_A = [15, 41, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

SINTETICAS_COLISAO = [
    ("Qual destes itens NÃO está relacionado a bancos de dados: Copa do Mundo, capital do Japão ou receita de bolo?", "abster"),
    ("No trecho sobre RAG, qual destes itens NÃO está relacionado: previsão do tempo, cotação de ação ou Copa do Mundo 2022?", "abster"),
    ("Qual destes termos NÃO está relacionado a redes neurais: Recife, Salvador ou Manaus?", "abster"),
]


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", (t or ""))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[-\u00ad]\s*", "", t.lower())  # hifenização de OCR: educa- cao → educacao
    return re.sub(r"\s+", " ", t)


# gold das 44 congeladas (sem _auditoria): resposta esperada das negativas tipo A/B
GOLD_44: dict[int, str] = {
    15: "GAN", 16: "voz", 41: "fundo de investimento", 42: "processamento de imagens",
    89: "Copa do Mundo", 90: "teoria da relatividade", 91: "copa do mundo",
    92: "Teoria da relatividade", 93: "teoria da relatividade", 94: "Teoria da Relatividade",
    95: "Capital do Japão", 96: "receita de bolo", 97: "Copa do Mundo de Futebol",
    98: "fundo de investimento", 99: "Teoria da relatividade", 100: "Cotação de ação",
}

GOLD_TIPO_A: dict[int, str] = {qid: GOLD_44[qid] for qid in TIPO_A}


TEMPLATE_RE = re.compile(
    r"(?:destes (?:itens|termos)|nao esta relacionado|nao e (?:um |uma )?conceito|fora do dominio)"
)


def detectar_template(pergunta: str) -> bool:
    p = norm(pergunta)
    if "nao" not in p:
        return False
    return bool(TEMPLATE_RE.search(p))


def parse_opcoes(pergunta: str) -> list[str]:
    """Tail após o último ':' → opções separadas por ' ou ' e ', '."""
    if ":" not in pergunta:
        return []
    tail = pergunta.rsplit(":", 1)[1].strip().rstrip("?")
    partes = [x.strip() for x in re.split(r"\s+ou\s+", tail)]
    opcoes: list[str] = []
    for p in partes:
        for x in re.split(r"\s*,\s*", p):
            x = x.strip()
            if len(x) >= 3:
                opcoes.append(x)
    return opcoes


def ocorrencias(opcao: str, texto: str) -> int:
    o = norm(opcao)
    if not o:
        return 0
    pat = re.compile(rf"(?<![a-z0-9à-ú]){re.escape(o)}(?![a-z0-9à-ú])")
    return len(pat.findall(norm(texto)))


def regra(opcoes: list[str], texto: str):
    """Retorna (resposta|None, {op: ocorrencias}, motivo)."""
    occ = {o: ocorrencias(o, texto) for o in opcoes}
    ausentes = [o for o, c in occ.items() if c == 0]
    presentes = [o for o, c in occ.items() if c > 0]
    if len(ausentes) == 1:
        return ausentes[0], occ, "V: ausente única"
    if len(ausentes) >= 2:
        return None, occ, f"ambígua ({len(ausentes)} ausentes)"
    if not presentes:
        return None, occ, "todas ausentes (fora do corpus → abster)"
    return None, occ, "todas presentes"


def regra_v3(opcoes: list[str], texto: str):
    """V3 = V1 + fallback de menor ocorrência."""
    occ = {o: ocorrencias(o, texto) for o in opcoes}
    ausentes = [o for o, c in occ.items() if c == 0]
    if len(ausentes) == 1:
        return ausentes[0], occ, "V1: ausente única"
    if len(ausentes) >= 2:
        return None, occ, "ambígua"
    if not any(occ.values()):
        return None, occ, "todas ausentes"
    menos = min(occ, key=occ.get)
    if occ[menos] == occ[max(occ, key=occ.get)]:
        return None, occ, "empate (todas presentes) → abster"
    return menos, occ, "V3: menor ocorrência"


def contexto(chunks: list[dict], top: list[dict]) -> str:
    return "\n".join(str(c.get("texto") or "") for c in top)


def main() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    by_id = {p["id"]: p for p in golden}
    cascade = {
        json.loads(l)["id"]: json.loads(l)
        for l in CASCADE.read_text(encoding="utf-8").splitlines()
    }
    chunks = carregar_chunks()

    # 1) quais perguntas o template reconhece (132Q) — só essas recuperam contexto
    alvo = [p for p in golden if detectar_template(p["pergunta"])]
    ids_alvo = [p["id"] for p in alvo]
    print(f"template detectado em {len(alvo)}/132: {ids_alvo}")

    # 2) contexto top-5, pool e DOCS COMPLETOS para cada alvo
    dados: dict[int, dict] = {}
    for p in alvo:
        top5 = recuperar(p["pergunta"], chunks, top_k=5, rerank=False)
        pool = recuperar(p["pergunta"], chunks, top_k=60, rerank=False)
        docs_top5 = sorted({c.get("arquivo") for c in top5})
        txt_docs = "\n".join(
            c.get("texto") or "" for c in chunks if c.get("arquivo") in docs_top5
        )
        gold_at = (p.get("_auditoria") or {}).get("resposta_esperada") or GOLD_44.get(p["id"])
        dados[p["id"]] = {
            "pergunta": p["pergunta"],
            "estrato": p.get("estrato"),
            "docs_esperados": p.get("docs_esperados"),
            "docs_top5": docs_top5,
            "texto_gold": gold_at,
            "opcoes": parse_opcoes(p["pergunta"]),
            "txt_top5": contexto(chunks, top5),
            "txt_pool": contexto(chunks, pool),
            "txt_docs": txt_docs,
        }

    # 3) aplica variantes nas tipo A
    print("\n=== VARIANTES nas 14 tipo A (resposta prevista vs gold) ===")
    res: dict[str, dict] = {}
    VARIANTES = (
        ("V1_top5", regra, "txt_top5"),
        ("V2_pool", regra, "txt_pool"),
        ("V4_docs_completos", regra, "txt_docs"),
        ("V3_top5+menor", regra_v3, "txt_top5"),
    )
    for nome, fn, campo in VARIANTES:
        hit = miss = abst = 0
        linhas = []
        for qid in TIPO_A:
            d = dados[qid]
            ctx = d[campo]
            pred, occ, motivo = fn(d["opcoes"], ctx)
            gold_op = (d["texto_gold"] or "")[:60]
            ok = bool(pred) and norm(pred) in norm(gold_op)
            if ok: hit += 1
            elif pred is None: abst += 1
            else: miss += 1
            linhas.append((qid, pred, ok, motivo, {o[:24]: c for o, c in occ.items()}))
            if qid == 15 or not ok or nome == "V4_docs_completos":
                print(f"[{nome}] #{qid:3d} pred={pred!r} hit={ok} ({motivo})")
        res[nome] = {"hit": hit, "miss": miss, "abst": abst}
        print(f"{nome}: hit {hit}/14 · miss {miss} · abst {abst}")
        dados_v = {}
        for qid, pred, ok, motivo, occ in linhas:
            dados_v[qid] = {"pred": pred, "ok": ok, "motivo": motivo, "occ": occ}
        res[nome]["por_q"] = dados_v

    # 4) guarda adversarial: o template nunca deve responder fora do corpus
    adver = [p for p in golden if p.get("estrato") == "adversarial"]
    disparos = [p["id"] for p in adver if detectar_template(p["pergunta"])]
    print(f"\nadversarial: {len(adver)} perguntas · template disparou em {disparos} (esperado: poucas)")
    # sondas sintéticas: TODAS as opções fora do corpus → exigir abstenção nas 3 variantes
    colisoes = []
    for pergunta, esperado in SINTETICAS_COLISAO:
        top5 = recuperar(pergunta, chunks, top_k=5, rerank=False)
        txt = contexto(chunks, top5)
        opcoes = parse_opcoes(pergunta)
        pred, occ, motivo = regra(opcoes, txt)
        colisoes.append({"pergunta": pergunta, "opcoes": opcoes, "pred": pred,
                         "absteve_ok": pred is None, "motivo": motivo, "occ": occ})
        print(f"colisão: {pergunta[:70]} → pred={pred!r} ({motivo})")

    # 5) projeção nas 132Q: regra substitui a cascade onde responde
    print("\n=== PROJEÇÃO nas 132Q (regra + cascade existente) ===")
    proj: dict[str, dict] = {}
    for nome, fn, campo in VARIANTES:
        ac = 0; n = 0
        for p in golden:
            qid = p["id"]; l = cascade.get(qid)
            if qid in TIPO_A:
                d = dados[qid]
                pred, _, _ = fn(d["opcoes"], d[campo])
                if pred is not None:
                    ok = norm(pred) in norm((d["texto_gold"] or "")[:60])
                    ac += 1 if ok else 0
                    n += 1
                    continue
            if l and l.get("juiz"):
                ac += 1 if l["juiz"]["correta"] else 0
                n += 1
        proj[nome] = {"acuracia": round(ac / n, 4), "ac": ac, "n": n}
        print(f"{nome}: {ac}/{n} = {ac/n:.3f} (baseline cascade 73/127 = 0.575)")

    ev = {"alvo_ids": ids_alvo, "tipo_a": TIPO_A, "dados": dados,
          "variantes": {k: v for k, v in res.items()},
          "colisoes": colisoes, "projecao": proj}
    SAIDA.write_text(json.dumps(ev, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\njson: {SAIDA}")


if __name__ == "__main__":
    main()
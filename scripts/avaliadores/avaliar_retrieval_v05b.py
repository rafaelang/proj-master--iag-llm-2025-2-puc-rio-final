"""v0.5b · P1 — avaliador ISOLADO de retrieval no golden expandido (sem LLM).

Mede Recalli@5 e MRR do retriever (BM25 + fastembed + RRF + RERANK opcional) sobre
o golden v0.5b (`perguntas_v05b.json`), com e sem o estágio de rerank — o degrau
RAG que degradou no corpus denso (recall@5 1.000 na v0.2 → 0.622 na v0.5b).

Estende `avaliar_retrieval.py` (v0.2) parametrizando o golden set e tornando a
diferenca rerank on/off o objetivo central do relatorio.

Uso:
  RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_retrieval_v05b.py
  RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_retrieval_v05b.py --no-evidencias
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from projeto_final import config
from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.retrieve import recuperar

GOLDEN = config.RAG_GOLDEN_SET
RESULTADOS_JSON = config.RAG_DIR / "retrieval_v05b.json"
EVIDENCIA_MD = config.DOCS_DIR / "v05b_retrieval.md"

TOP_K = 5


def _recall_por_pergunta(docs_esperados: set[str], docs_rec: list[str]) -> float:
    if not docs_esperados:
        return 0.0
    return len(docs_esperados & set(docs_rec)) / len(docs_esperados)


def _mrr_por_pergunta(docs_esperados: set[str], top: list[dict]) -> float:
    for i, c in enumerate(top, start=1):
        if c["arquivo"] in docs_esperados:
            return 1.0 / i
    return 0.0


def avaliar(top_k: int = TOP_K, rerank: bool = True) -> dict:
    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    chunks = carregar_chunks()

    logger.info("Golden {}: {} perguntas · {} chunks · rerank={}", GOLDEN.name, len(perguntas), len(chunks), rerank)
    linhas = []
    for q in perguntas:
        esperados = set(q.get("docs_esperados", []))
        top = recuperar(q["pergunta"], chunks, top_k=top_k, rerank=rerank)
        linhas.append({
            "id": q["id"],
            "estrato": q["estrato"],
            "pergunta": q["pergunta"],
            "docs_esperados": sorted(esperados),
            "recall@5": _recall_por_pergunta(esperados, [c["arquivo"] for c in top]),
            "mrr": _mrr_por_pergunta(esperados, top),
            "docs_recuperados": [c["arquivo"] for c in top],
        })

    com_docs = [l for l in linhas if l["docs_esperados"]]

    def media(key: str) -> float | None:
        return round(sum(l[key] for l in com_docs) / len(com_docs), 3) if com_docs else None

    resumo: dict = {
        "n_perguntas": len(linhas),
        "n_com_docs_esperados": len(com_docs),
        "recall@5": media("recall@5"),
        "mrr": media("mrr"),
        "por_estrato": {},
    }
    for est in ("rotineira", "composta", "negativa", "adversarial"):
        le = [l for l in com_docs if l["estrato"] == est]
        if not le:
            continue
        resumo["por_estrato"][est] = {
            "n": len(le),
            "recall@5": round(sum(l["recall@5"] for l in le) / len(le), 3),
            "mrr": round(sum(l["mrr"] for l in le) / len(le), 3),
        }
    return {"resumo": resumo, "perguntas": linhas}


def gerar_evidencia(sem: dict, com: dict) -> str:
    s, c = sem["resumo"], com["resumo"]
    md = [
        "# v0.5b · P1 — Retrieval isolado no golden expandido (recall@5)",
        "",
        f"> Dataset: `{GOLDEN.name}` ({s['n_perguntas']} perguntas, "
        f"{s['n_com_docs_esperados']} com `docs_esperados`).",
        "> Retriever: BM25 próprio + fastembed + RRF (BM25 1.0 × denso 1.5, k=60, imagem 1.15)",
        f"> RERANK: jina-reranker-v2-base-multilingual · pool over-fetch {retrieve.POOL_RERANK} · top_k {TOP_K}.",
        "",
        "## Resultado — RERANK on × off",
        "",
        "| Métrica | RRF direto (sem rerank) | RRF + RERANK | Δ |",
        "|---|---|---|---|",
        f"| **Recall@5** | **{s['recall@5']:.3f}** | **{c['recall@5']:.3f}** | {c['recall@5']-s['recall@5']:+.3f} |",
        f"| MRR | {s['mrr']:.3f} | {c['mrr']:.3f} | {c['mrr']-s['mrr']:+.3f} |",
        "",
        "Por estrato (recall@5):",
        "",
    ]
    estratos = sorted(set(list(s["por_estrato"]) + list(c["por_estrato"])))
    md.append("| Estrato | n | sem rerank | com rerank | Δ |")
    md.append("|---|---|---|---|---|")
    for e in estratos:
        ms, mc = s["por_estrato"].get(e, {}), c["por_estrato"].get(e, {})
        n = ms.get("n") or mc.get("n")
        rs = ms.get("recall@5", 0)
        rc = mc.get("recall@5", 0)
        md.append(f"| {e} | {n} | {rs:.3f} | {rc:.3f} | {rc-rs:+.3f} |")
    md += [
        "",
        "## Perguntas que o RERANK resgatou (miss sem rerank → hit com rerank)",
        "",
    ]
    p_sem = {p["id"]: p for p in sem["perguntas"]}
    resgatadas = [
        p for p in com["perguntas"]
        if p["docs_esperados"] and p["recall@5"] > 0 and p_sem[p["id"]]["recall@5"] == 0
    ]
    if resgatadas:
        for p in resgatadas:
            md.append(f"- **#{p['id']:02d}** [{p['estrato']}] {p['pergunta'][:80]}")
    else:
        md.append("- (nenhuma)")
    md += [
        "",
        "## Misses (sem rerank e/ou com rerank)",
        "",
    ]
    for p in com["perguntas"]:
        if not p["docs_esperados"]:
            continue
        rotulos = ("sem rerank" if p_sem[p["id"]]["recall@5"] == 0 else "",
                   "com rerank" if p["recall@5"] == 0 else "")
        falha = [r for r in rotulos if r]
        if falha:
            md.append(f"- **#{p['id']:02d}** [{p['estrato']}] ({', '.join(falha)}) "
                      f"{p['pergunta'][:70]} → esperado {p['docs_esperados']}")
    md += [
        "",
        "Detalhes por pergunta em `data/processed/rag/retrieval_v05b.json`.",
        "",
    ]
    return "\n".join(md) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliador de retrieval v0.5b (RERANK on × off).")
    parser.add_argument("--no-evidencia", action="store_true", help="nao escreve docs/v05b_retrieval.md")
    args = parser.parse_args()

    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    sem = avaliar(rerank=False)
    com = avaliar(rerank=True)
    RESULTADOS_JSON.write_text(json.dumps({"sem_rerank": sem, "com_rerank": com},
                                          ensure_ascii=False, indent=2), encoding="utf-8")

    s, c, = sem["resumo"], com["resumo"]
    print("=== RETRIEVAL v0.5b (sem LLM) ===")
    print(f"perguntas: {s['n_perguntas']} (com docs: {s['n_com_docs_esperados']})")
    print(f"Recall@5  sem rerank: {s['recall@5']:.3f}  |  com rerank: {c['recall@5']:.3f}")
    print(f"MRR       sem rerank: {s['mrr']:.3f}  |  com rerank: {c['mrr']:.3f}")

    if not args.no_evidencia:
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        EVIDENCIA_MD.write_text(gerar_evidencia(sem, com), encoding="utf-8")
        print(f"evidência: {EVIDENCIA_MD}")
    print(f"json: {RESULTADOS_JSON}")


if __name__ == "__main__":
    main()
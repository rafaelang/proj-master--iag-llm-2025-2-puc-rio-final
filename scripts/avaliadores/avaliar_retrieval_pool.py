"""Avaliador de retrieval v0.5b parametrizado por POOL (rerank ligado) — rodar no Colab.

Rodada de decisao P1-combinado: mede recall@5 e MRR do retriever REAL de
producao (BM25 + fastembed + RRF ponderado + DEDUP + RERANK cross-encoder)
variando o pool de over-fetch (POOL_RERANK) em 30/60/100/200/400.

Uso (Colab / CPU):
    POOL=60 python avaliar_retrieval_pool.py --no-evidencia
Salva por pool em data/processed/rag/retrieval_pool_{pool}.json (1 bloco com_rerank).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import projeto_final.rag.retrieve as RETRIEVE

from projeto_final import config
from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.retrieve import recuperar

GOLDEN = config.RAG_GOLDEN_SET
TOP_K = 5


def _recall(docs_esperados: set[str], docs_rec: list[str]) -> float:
    if not docs_esperados:
        return 0.0
    return len(docs_esperados & set(docs_rec)) / len(docs_esperados)


def _mrr(docs_esperados: set[str], top: list[dict]) -> float:
    for i, c in enumerate(top, start=1):
        if c["arquivo"] in docs_esperados:
            return 1.0 / i
    return 0.0


def avaliar(pool: int) -> dict:
    RETRIEVE.POOL_RERANK = pool
    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    chunks = carregar_chunks()

    linhas = []
    for q in perguntas:
        esperados = set(q.get("docs_esperados", []))
        top = recuperar(q["pergunta"], chunks, top_k=TOP_K, rerank=True)
        linhas.append({
            "id": q["id"],
            "estrato": q["estrato"],
            "pergunta": q["pergunta"],
            "docs_esperados": sorted(esperados),
            "recall@5": _recall(esperados, [c["arquivo"] for c in top]),
            "mrr": _mrr(esperados, top),
            "docs_recuperados": [c["arquivo"] for c in top],
        })

    com_docs = [l for l in linhas if l["docs_esperados"]]

    def media(key: str):
        return round(sum(l[key] for l in com_docs) / len(com_docs), 3) if com_docs else None

    resumo = {
        "pool": pool,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--no-evidencia", action="store_true")
    args = parser.parse_args()

    pools = [args.pool] if args.pool else [30, 60, 100, 200, 400]
    for pool in pools:
        r = avaliar(pool)
        s = r["resumo"]
        print("=== POOL %d | recall@5 %.3f | MRR %.3f | %d/%d ===" % (
            pool, s["recall@5"] or 0, s["mrr"] or 0, round((s["recall@5"] or 0) * s["n_com_docs_esperados"]), s["n_com_docs_esperados"]))
        for est, v in sorted(s["por_estrato"].items()):
            print("   %-10s recall@5 %.3f MRR %.3f (%d)" % (est, v["recall@5"], v["mrr"], v["n"]))
        out = config.RAG_DIR / f"retrieval_pool_{pool}.json"
        out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   -> {out}")


if __name__ == "__main__":
    main()
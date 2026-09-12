"""Variacao de pool SEM rerank no retriever real de producao (dedup + RRF).

Mede recall@5/MRR replicando `recuperar` (BM25 + denso + RRF ponderado + dedup)
parametrizando o over-fetch de candidatos, sem cross-encoder. Em producao o
over-fetch e `POOL_RERANK` (com rerank) ou `top_k*2` (sem rerank) — aqui forcamos
o pool nos dois caminhos para medir o efeito puro do pool.

Uso (Colab): python avaliar_retrieval_pool_norerank.py --pools 10,60,100,200
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/content/src")
os.environ.setdefault("RAG_GOLDEN_SET_FILE", "perguntas_v05b.json")

from projeto_final import config
from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.retrieve import (
    K_RRF,
    PESO_BM25,
    PESO_DENSO,
    PESO_IMAGEM_RRF,
    _ranking_bm25,
    _ranking_embeddings,
    _deduplicar,
)

GOLDEN = config.RAG_GOLDEN_SET
TOP_K = 5


def _recuperar_pool(pergunta, chunks, bm25, embeddings, chunk_ids, pool):
    r_b = _ranking_bm25(pergunta, bm25, chunks, top_k=pool)
    r_d = _ranking_embeddings(pergunta, embeddings, chunk_ids, chunks, top_k=pool)
    tipo = {c["id"]: c.get("tipo", "texto") for c in chunks}
    scores = {}
    for cid in set(r_b) | set(r_d):
        s = PESO_BM25 * r_b.get(cid, 0.0) + PESO_DENSO * r_d.get(cid, 0.0)
        if tipo.get(cid) == "imagem":
            s *= PESO_IMAGEM_RRF
        scores[cid] = s
    ordem = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ch_map = {c["id"]: c for c in chunks}
    candidatos = [ch_map[cid] for cid, _ in ordem[:pool] if cid in ch_map]
    candidatos = _deduplicar(candidatos)
    return candidatos[:TOP_K]


def _recall(docs_esperados, docs_rec):
    if not docs_esperados:
        return 0.0
    return len(docs_esperados & set(docs_rec)) / len(docs_esperados)


def _mrr(docs_esperados, top):
    for i, c in enumerate(top, start=1):
        if c["arquivo"] in docs_esperados:
            return 1.0 / i
    return 0.0


def avaliar(pool):
    from projeto_final.rag.index import carregar_indices
    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    chunks = carregar_chunks()
    bm25, embeddings, chunk_ids = carregar_indices()
    linhas = []
    for q in perguntas:
        esperados = set(q.get("docs_esperados", []))
        top = _recuperar_pool(q["pergunta"], chunks, bm25, embeddings, chunk_ids, pool)
        linhas.append({
            "id": q["id"], "estrato": q["estrato"], "pergunta": q["pergunta"],
            "docs_esperados": sorted(esperados),
            "recall@5": _recall(esperados, [c["arquivo"] for c in top]),
            "mrr": _mrr(esperados, top),
            "docs_recuperados": [c["arquivo"] for c in top],
        })
    com_docs = [l for l in linhas if l["docs_esperados"]]
    media = lambda k: round(sum(l[k] for l in com_docs) / len(com_docs), 3) if com_docs else None
    resumo = {"pool": pool, "n_perguntas": len(linhas), "n_com_docs_esperados": len(com_docs),
              "recall@5": media("recall@5"), "mrr": media("mrr"), "por_estrato": {}}
    for est in ("rotineira", "composta", "negativa", "adversarial"):
        le = [l for l in com_docs if l["estrato"] == est]
        if le:
            resumo["por_estrato"][est] = {"n": len(le),
                                          "recall@5": round(sum(l["recall@5"] for l in le) / len(le), 3),
                                          "mrr": round(sum(l["mrr"] for l in le) / len(le), 3)}
    return {"resumo": resumo, "perguntas": linhas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="10,60,100,200")
    a = ap.parse_args()
    pools = [int(p) for p in a.pools.split(",")]
    for pool in pools:
        r = avaliar(pool)
        s = r["resumo"]
        print("=== POOL %d | recall@5 %.3f | MRR %.3f | %d/%d ===" % (
            pool, s["recall@5"] or 0, s["mrr"] or 0,
            round((s["recall@5"] or 0) * s["n_com_docs_esperados"]), s["n_com_docs_esperados"]))
        for est, v in sorted(s["por_estrato"].items()):
            print("   %-10s recall@5 %.3f MRR %.3f (%d)" % (est, v["recall@5"], v["mrr"], v["n"]))
        out = config.RAG_DIR / f"recall_pool_{pool}_norerank.json"
        out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print("   ->", out)


if __name__ == "__main__":
    main()
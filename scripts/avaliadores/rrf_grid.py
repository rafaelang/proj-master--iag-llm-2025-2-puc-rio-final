"""Grid de pesos RRF (BM25 x denso x K) no pool 60 — sem rerank, golden v0.5b corrigido."""
import sys, os, json
sys.path.insert(0, "/content/src")
os.environ["RAG_GOLDEN_SET_FILE"] = "perguntas_v05b.json"
from projeto_final import config
from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.index import carregar_indices
from projeto_final.rag.retrieve import _ranking_bm25, _ranking_embeddings, _deduplicar

GOLDEN = config.RAG_GOLDEN_SET
POOL = 60
TOP_K = 5


def rec_pool(pergunta, chunks, bm25, embeddings, cids, wb, wd, k_rrf, pool):
    r_b = _ranking_bm25(pergunta, bm25, chunks, top_k=pool)
    r_d = _ranking_embeddings(pergunta, embeddings, cids, chunks, top_k=pool)
    scores = {}
    for cid in set(r_b) | set(r_d):
        scores[cid] = wb * r_b.get(cid, 0.0) + wd * r_d.get(cid, 0.0)
    ordem = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    cmap = {c["id"]: c for c in chunks}
    cands = [cmap[cid] for cid, _ in ordem[:pool] if cid in cmap]
    cands = _deduplicar(cands)
    return [c["arquivo"] for c in cands[:TOP_K]]


def main():
    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    chunks = carregar_chunks()
    bm25, embeddings, cids = carregar_indices()
    combos = [(wb, wd) for wb in (0.5, 1.0, 1.5) for wd in (1.0, 1.25, 1.5, 2.0, 3.0)]
    resultados = []
    for wb, wd in combos:
        hits = 0; tot = 0; mrr = 0.0
        for q in perguntas:
            exp = set(q.get("docs_esperados", []))
            if not exp:
                continue
            tot += 1
            rec = rec_pool(q["pergunta"], chunks, bm25, embeddings, cids, wb, wd, 60, POOL)
            inter = exp & set(rec)
            hits += 1 if inter else 0
            for i, f in enumerate(rec, 1):
                if f in exp:
                    mrr += 1.0 / i; break
        resultados.append((wb, wd, round(hits / tot, 3), round(mrr / tot, 3), hits, tot))
    resultados.sort(key=lambda r: -r[2])
    print("wb   wd   recall@5  MRR   hits")
    for wb, wd, r5, m, hits, tot in resultados[:8]:
        print(f"{wb:4.1f} {wd:4.2f}  {r5:.3f}   {m:.3f}  {hits}/{tot}")
    json.dump(resultados, open("/content/rrf_grid.json", "w"), indent=2)
    print("salvo /content/rrf_grid.json")


if __name__ == "__main__":
    main()
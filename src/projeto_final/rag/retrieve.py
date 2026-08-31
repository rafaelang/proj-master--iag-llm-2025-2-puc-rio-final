"""Recuperacao hibrida: BM25 + embeddings + RRF."""

from __future__ import annotations

import numpy as np
from loguru import logger

from projeto_final import config
from projeto_final.rag.index import carregar_indices, construir_indices, _tokenizar

K_RRF = 60  # constante padrao do RRF


def _normalizar(vetores: np.ndarray) -> np.ndarray:
    """Normaliza vetores para cosine similarity."""
    norm = np.linalg.norm(vetores, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return vetores / norm


def _ranking_bm25(pergunta: str, bm25, chunks: list[dict], top_k: int = 50) -> dict[int, float]:
    tokens = _tokenizar(pergunta)
    scores = bm25.get_scores(tokens)
    indices = np.argsort(scores)[::-1][:top_k]
    return {chunks[i]["id"]: 1.0 / (rank + 1 + K_RRF) for rank, i in enumerate(indices) if scores[i] > 0}


def _ranking_embeddings(pergunta: str, embeddings: np.ndarray, chunk_ids: list[int], chunks: list[dict], top_k: int = 50) -> dict[int, float]:
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=str(config.RAG_DIR / "embed_models"))
    query_vec = np.array(list(model.embed([pergunta])))[0]
    query_vec = query_vec / np.linalg.norm(query_vec)
    embeddings_norm = _normalizar(embeddings)
    scores = embeddings_norm @ query_vec
    indices = np.argsort(scores)[::-1][:top_k]
    return {chunk_ids[i]: 1.0 / (rank + 1 + K_RRF) for rank, i in enumerate(indices)}


def recuperar(pergunta: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Recupera top-k chunks usando fusao RRF de BM25 + embeddings."""
    bm25, embeddings, chunk_ids = construir_indices(chunks)
    r_bm25 = _ranking_bm25(pergunta, bm25, chunks)
    r_emb = _ranking_embeddings(pergunta, embeddings, chunk_ids, chunks)

    # RRF
    scores = {}
    for cid in set(r_bm25) | set(r_emb):
        scores[cid] = r_bm25.get(cid, 0.0) + r_emb.get(cid, 0.0)

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in ranking[:top_k]]
    logger.debug("Recuperacao RRF: pergunta='{}' -> top {} ids={}", pergunta, top_k, top_ids)

    chunk_map = {c["id"]: c for c in chunks}
    return [chunk_map[cid] for cid in top_ids if cid in chunk_map]

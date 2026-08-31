"""Indexacao BM25 + embeddings para RAG."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from loguru import logger
from rank_bm25 import BM25Okapi

from projeto_final import config


def _tokenizar(texto: str) -> list[str]:
    """Tokenizer simples para BM25: minusculas, sem acento, sem pontuacao."""
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [tok for tok in t.split() if tok]


def construir_indices(chunks: list[dict], force: bool = False) -> tuple[BM25Okapi, np.ndarray, list[int]]:
    """Constroi BM25 e embeddings para uma lista de chunks.

    Retorna (bm25, embeddings, chunk_ids)."""
    if not force and config.RAG_BM25_PATH.exists() and config.RAG_EMBEDDINGS_PATH.exists():
        return carregar_indices()

    logger.info("Construindo indices RAG para {} chunks...", len(chunks))
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)

    # BM25
    corpus = [chunk["texto"] for chunk in chunks]
    tokenizado = [_tokenizar(c) for c in corpus]
    bm25 = BM25Okapi(tokenizado)
    with open(config.RAG_BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    # Embeddings
    model = TextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=str(config.RAG_DIR / "embed_models"))
    embeddings = np.array(list(model.embed(corpus)))
    np.save(config.RAG_EMBEDDINGS_PATH, embeddings)

    chunk_ids = [chunk["id"] for chunk in chunks]
    config.RAG_CHUNK_IDS_PATH.write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")

    logger.info("Indices salvos: BM25={}, embeddings shape={}", config.RAG_BM25_PATH, embeddings.shape)
    return bm25, embeddings, chunk_ids


def carregar_indices() -> tuple[BM25Okapi, np.ndarray, list[int]]:
    """Carrega indices BM25 e embeddings previamente construidos."""
    logger.debug("Carregando indices RAG de disco...")
    with open(config.RAG_BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    embeddings = np.load(config.RAG_EMBEDDINGS_PATH)
    chunk_ids = json.loads(config.RAG_CHUNK_IDS_PATH.read_text(encoding="utf-8"))
    return bm25, embeddings, chunk_ids

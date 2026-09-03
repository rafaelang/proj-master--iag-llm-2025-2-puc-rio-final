"""Indexacao BM25 + embeddings para RAG.

A v0.3 (texto + imagem) usa o mesmo tipo de indice em uma PASTA propria
(RAG_V3_DIR), parametrizada por `base`; sem `base`, o comportamento e identico
ao da v0.2 (RAG_DIR, somente texto).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from loguru import logger

from projeto_final import config
from projeto_final.bm25 import BM25Okapi, normalizar


def _caminhos(base: Path | None):
    """Resolve os caminhos dos artefatos de indice a partir de uma pasta base."""
    base = base or config.RAG_DIR
    return {
        "bm25": base / "bm25.pkl",
        "embeddings": base / "embeddings.npy",
        "chunk_ids": base / "chunk_ids.json",
    }


def construir_indices(
    chunks: list[dict], force: bool = False, base: Path | None = None
) -> tuple[BM25Okapi, np.ndarray, list[int]]:
    """Constroi BM25 e embeddings para uma lista de chunks.

    Retorna (bm25, embeddings, chunk_ids)."""
    cam = _caminhos(base)
    if not force and cam["bm25"].exists() and cam["embeddings"].exists():
        return carregar_indices(base)

    logger.info("Construindo indices para {} chunks (base={})...", len(chunks), cam["bm25"].parent)
    cam["bm25"].parent.mkdir(parents=True, exist_ok=True)

    # BM25
    corpus = [chunk["texto"] for chunk in chunks]
    tokenizado = [normalizar(c) for c in corpus]
    bm25 = BM25Okapi(tokenizado)
    with open(cam["bm25"], "wb") as f:
        pickle.dump(bm25, f)

    # Embeddings
    model = TextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=str(config.RAG_DIR / "embed_models"))
    embeddings = np.array(list(model.embed(corpus)))
    np.save(cam["embeddings"], embeddings)

    chunk_ids = [chunk["id"] for chunk in chunks]
    cam["chunk_ids"].write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")

    logger.info("Indices salvos: BM25={}, embeddings shape={}", cam["bm25"], embeddings.shape)
    return bm25, embeddings, chunk_ids


def carregar_indices(base: Path | None = None) -> tuple[BM25Okapi, np.ndarray, list[int]]:
    """Carrega indices BM25 e embeddings previamente construidos."""
    cam = _caminhos(base)
    logger.debug("Carregando indices de disco (base={})...", cam["bm25"].parent)
    with open(cam["bm25"], "rb") as f:
        bm25 = pickle.load(f)
    embeddings = np.load(cam["embeddings"])
    chunk_ids = json.loads(cam["chunk_ids"].read_text(encoding="utf-8"))
    return bm25, embeddings, chunk_ids

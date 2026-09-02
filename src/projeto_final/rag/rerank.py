"""Rerank dos candidatos recuperados com cross-encoder (fastembed ONNX).

Modelo: jinaai/jina-reranker-v2-base-multilingual (multilingue, ideal para pt-BR).
O modelo e carregado de forma LAZY e em singleton (uma unica instancia por processo).
"""

from __future__ import annotations

from loguru import logger

from projeto_final import config

_reranker_cache = None


def _get_reranker():
    """Retorna a instancia unica do cross-encoder (carrega o modelo sob demanda)."""
    global _reranker_cache
    if _reranker_cache is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        logger.info("Carregando reranker {} ...", config.RERANK_MODEL)
        _reranker_cache = TextCrossEncoder(
            model_name=config.RERANK_MODEL,
            cache_dir=str(config.RERANK_CACHE_DIR),
        )
        logger.info("Reranker carregado: {}", config.RERANK_MODEL)
    return _reranker_cache


def rerank_chunks(pergunta: str, candidatos: list[dict]) -> list[dict]:
    """Reordena os chunks candidatos por score do cross-encoder (decrescente)."""
    if not candidatos:
        return candidatos
    reranker = _get_reranker()
    textos = [c["texto"] for c in candidatos]
    scores = list(reranker.rerank(pergunta, textos))
    ordenados = sorted(zip(candidatos, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ordenados]

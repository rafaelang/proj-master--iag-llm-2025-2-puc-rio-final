"""Recuperacao hibrida: BM25 + embeddings + RRF."""

from __future__ import annotations

import numpy as np
from loguru import logger

from projeto_final import config
from projeto_final.bm25 import normalizar, preparar_query
from projeto_final.rag.index import carregar_indices, construir_indices
from projeto_final.rag.rerank import rerank_chunks

K_RRF = 60  # constante padrao do RRF
# Pesos da fusao RRF: denso pesa mais (semantica ajuda mais em perguntas "o que e X").
PESO_BM25 = 1.0
PESO_DENSO = 1.5
# v0.3: peso extra para chunks de IMAGEM no RRF. Sem ele, listas de rotulos OCR
# (figuras) quase nunca superam a prosa do mesmo slide na disputa pelo top-5.
PESO_IMAGEM_RRF = 1.15
# Candidatos (por lado) buscados antes da fusao + rerank quando o rerank esta ativo.
POOL_RERANK = 30
# Dedup: Jaccard minimo entre dois chunks da MESMA pagina/doc para tratar como
# duplicata (overlay de chunking e prefixos de imagem duplicam conteudo).
DEDUP_JACCARD = 0.85


def _deduplicar(candidatos: list[dict]) -> list[dict]:
    """Remove chunks duplicados e quase-duplicados do retriever.

    - duplicata EXATA: mesmo documento e mesmo texto normalizado (slides/trechos
      repetidos no mesmo PDF);
    - quase-duplicata: mesmo documento+pagina em que um chunk esta (quase)
      contido no outro (>= DEDUP_JACCARD de sobreposicao via contencao) — caso
      tipico do overlay de chunking e do prefixo de pagina nos chunks de imagem.

    Mantem o chunk de maior score (primeiro da lista, ordenada pelo RRF/rerank).
    Chunks identicos em documentos DIFERENTES sao mantidos (cada um e uma fonte).
    """
    saida: list[dict] = []
    chaves_exatas: set[tuple] = set()
    for c in candidatos:
        doc = c.get("doc_id")
        toks = tuple(normalizar(c["texto"]))
        chave = (doc, toks)
        if chave in chaves_exatas:
            continue
        descartar = False
        if doc is not None and toks:
            conjunto = set(toks)
            for aceito in saida:
                if aceito.get("doc_id") != doc:
                    continue
                aceito_toks = set(normalizar(aceito["texto"]))
                if not aceito_toks:
                    continue
                # contencao: fracao do menor conjunto que esta no maior
                inter = len(conjunto & aceito_toks)
                contencao = inter / min(len(conjunto), len(aceito_toks))
                mesma_pagina = aceito.get("pagina") == c.get("pagina")
                if mesma_pagina and contencao >= DEDUP_JACCARD:
                    descartar = True
                    break
        if not descartar:
            chaves_exatas.add(chave)
            saida.append(c)
    removidos = len(candidatos) - len(saida)
    if removidos:
        logger.debug("Dedup: {} candidatos duplicados removidos", removidos)
    return saida


def _normalizar(vetores: np.ndarray) -> np.ndarray:
    """Normaliza vetores para cosine similarity."""
    norm = np.linalg.norm(vetores, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return vetores / norm


def _ranking_bm25(pergunta: str, bm25, chunks: list[dict], top_k: int = 50) -> dict[int, float]:
    tokens = preparar_query(pergunta)
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


def recuperar(
    pergunta: str,
    chunks: list[dict],
    top_k: int = 5,
    rerank: bool = False,
    base=None,
) -> list[dict]:
    """Recupera top-k chunks: busca hibrida BM25 + denso, fusao RRF ponderada e RERANK opcional.

    Fluxo:
      1) BM25 e denso buscam candidatos (over-fetch: POOL_RERANK=30 com rerank,
         top_k*2 sem rerank) -> RRF ponderado (k=60, BM25 1.0 x denso 1.5);
      2) RERANK opcional dos candidatos com cross-encoder (rag/rerank.py);
      3) retorna estritamente top_k (o pipeline usa top_k=5 -> Top-5 definitivo).

    `base` seleciona a pasta do indice persistido (None = v0.2 texto; a v0.3 usa
    config.RAG_V3_DIR). rerank esta DESATIVADO por padrao (medicao: MRR 0.906 ->
    0.865 no golden set; custo ~40 s/query em CPU). Use rerank=True para ativa-lo.
    """
    bm25, embeddings, chunk_ids = construir_indices(chunks, base=base)
    candidatos_n = POOL_RERANK if rerank else top_k * 2
    r_bm25 = _ranking_bm25(pergunta, bm25, chunks, top_k=candidatos_n)
    r_emb = _ranking_embeddings(pergunta, embeddings, chunk_ids, chunks, top_k=candidatos_n)

    # RRF ponderado (+ peso de imagem da v0.3, quando o corpus tem figuras)
    tipo_por_id = {c["id"]: c.get("tipo", "texto") for c in chunks}
    scores = {}
    for cid in set(r_bm25) | set(r_emb):
        s = PESO_BM25 * r_bm25.get(cid, 0.0) + PESO_DENSO * r_emb.get(cid, 0.0)
        if tipo_por_id.get(cid) == "imagem":
            s *= PESO_IMAGEM_RRF
        scores[cid] = s

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in ranking[:candidatos_n]]
    logger.debug("Recuperacao RRF: pergunta='{}' -> {} candidatos ids={}", pergunta, len(top_ids), top_ids)

    chunk_map = {c["id"]: c for c in chunks}
    candidatos = [chunk_map[cid] for cid in top_ids if cid in chunk_map]
    # Dedup antes do rerank/entrega: remove duplicatas de texto e quase-duplicatas
    # (mesma pagina/doc com alta sobreposicao) mantendo o chunk de maior score.
    candidatos = _deduplicar(candidatos)

    if rerank:
        candidatos = rerank_chunks(pergunta, candidatos)
        logger.info("Rerank aplicado: {} candidatos -> top-{}", len(candidatos), top_k)
    else:
        logger.debug("Rerank desativado: RRF direto -> top {}\n", top_k)

    return candidatos[:top_k]

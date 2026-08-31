"""Pipeline RAG: pergunta -> chunks -> resposta com citacao."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from projeto_final import config
from projeto_final.rag.chunk import chunk_paginas
from projeto_final.rag.ingest import ingest, salvar_paginas
from projeto_final.rag.retrieve import recuperar
from projeto_final import llm as llm_mod


def carregar_chunks() -> list[dict]:
    """Carrega chunks do disco; constroi a partir do corpus se nao existir."""
    if config.RAG_CHUNK_PATH.exists():
        logger.debug("Carregando chunks de {}", config.RAG_CHUNK_PATH)
        return json.loads(config.RAG_CHUNK_PATH.read_text(encoding="utf-8"))

    logger.info("Chunks nao encontrados. Ingestionando corpus...")
    paginas = ingest(config.RAW_DIR)
    salvar_paginas(paginas, config.RAG_DIR / "paginas.json")
    chunks = chunk_paginas(paginas)
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    config.RAG_CHUNK_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("{} chunks salvos em {}", len(chunks), config.RAG_CHUNK_PATH)
    return chunks


def _formatar_contexto(chunks: list[dict]) -> str:
    linhas = []
    for i, chunk in enumerate(chunks, start=1):
        linhas.append(f"[{i}] {chunk['doc_id']}, pagina {chunk['pagina']}: {chunk['texto']}")
    return "\n\n".join(linhas)


def responder(pergunta: str, top_k: int = 10) -> dict:
    """Executa o pipeline RAG completo e retorna dict com resposta, chunks e metadados."""
    t0 = __import__("time").time()
    chunks = carregar_chunks()
    recuperados = recuperar(pergunta, chunks, top_k=top_k)
    contexto = _formatar_contexto(recuperados)
    sistema = config.ler_prompt("v0.2/rag_sistema.txt")
    resposta, meta_llm = llm_mod.responder_com_contexto(pergunta, contexto, sistema=sistema)
    abstencao = llm_mod.detectar_abstencao(resposta)
    logger.info("RAG: pergunta='{}' abstencao={} chunks={}", pergunta, abstencao, len(recuperados))
    return {
        "pergunta": pergunta,
        "resposta": resposta,
        "abstencao": abstencao,
        "chunks": recuperados,
        "contexto": contexto,
        "latencia_s": round(__import__("time").time() - t0, 2),
        "modelo_llm": meta_llm.get("modelo"),
        "uso": meta_llm.get("uso"),
    }


def reconstruir_indices() -> None:
    """Forca a reconstrucao dos indices a partir do corpus."""
    from projeto_final.rag.index import construir_indices
    chunks = carregar_chunks()
    construir_indices(chunks, force=True)

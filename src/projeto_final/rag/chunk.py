"""Chunking por fronteira natural para RAG."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger


def _limpar(texto: str) -> str:
    """Remove espacos e quebras de linha excessivas."""
    texto = re.sub(r"\n+", "\n", texto)
    texto = re.sub(r" +", " ", texto)
    return texto.strip()


def _dividir_em_frases(texto: str) -> list[str]:
    """Divide texto em frases respeitando pontuacao final."""
    padrao = r"(?<=[.!?])\s+"
    return [f.strip() for f in re.split(padrao, texto) if f.strip()]


def chunk_paginas(paginas: list[dict], tamanho_alvo: int = 300, sobreposicao: int = 50) -> list[dict]:
    """Cria chunks por fronteira natural (frases) a partir de paginas extraidas."""
    chunks = []
    chunk_id = 0
    for pag in paginas:
        texto = _limpar(pag["texto"])
        if not texto:
            continue
        frases = _dividir_em_frases(texto)
        atual = ""
        for frase in frases:
            if len(atual.split()) + len(frase.split()) <= tamanho_alvo:
                atual += " " + frase if atual else frase
            else:
                if atual:
                    chunk_id += 1
                    chunks.append(_criar_chunk(atual, pag, chunk_id))
                # inicia novo chunk com sobreposicao do anterior
                palavras_anterior = atual.split() if atual else []
                overlap = " ".join(palavras_anterior[-sobreposicao:]) if len(palavras_anterior) > sobreposicao else atual
                atual = (overlap + " " + frase).strip() if overlap else frase
        if atual:
            chunk_id += 1
            chunks.append(_criar_chunk(atual, pag, chunk_id))
    logger.info("Chunking completo: {} chunks", len(chunks))
    return chunks


def _criar_chunk(texto: str, pag: dict, chunk_id: int) -> dict:
    return {
        "id": chunk_id,
        "doc_id": pag["doc_id"],
        "arquivo": pag["arquivo"],
        "pagina": pag["pagina"],
        "titulo": pag.get("titulo"),
        "texto": texto.strip(),
        "tokens": len(texto.split()),
    }

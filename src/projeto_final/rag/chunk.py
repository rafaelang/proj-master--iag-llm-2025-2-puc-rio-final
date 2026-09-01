"""Chunking por fronteira natural (paragrafos) para RAG.

Estrategia (Terceira Rodada da v0.2):
- Fronteira principal: PARAGRAFO (linha em branco).
- Paragrafos acumulados ate max_chars (padrao 1200 chars).
- Paragrafo gigante (> max_chars) e quebrado em FRASES completas.
- doc_id do chunk = nome do arquivo COM extensao (ex. nlp_aula06_rag_avancado_ocr.pdf),
  casando exatamente com docs_esperados do golden set.
"""

from __future__ import annotations

import re

from loguru import logger


def _limpar(texto: str) -> str:
    """Limpa o texto preservando a fronteira de paragrafo (linha em branco)."""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)          # colapsa espacos/tabs dentro da linha
    texto = re.sub(r"\n{3,}", "\n\n", texto)       # colapsa multiplas linhas em branco em uma
    return texto.strip()


def _dividir_em_paragrafos(texto: str) -> list[str]:
    """Divide o texto em paragrafos por linha em branco."""
    return [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]


def _dividir_em_frases(texto: str) -> list[str]:
    """Divide um trecho em frases completas (pontuacao final)."""
    padrao = r"(?<=[.!?])\s+"
    return [f.strip() for f in re.split(padrao, texto) if f.strip()]


def _quebrar_paragrafo_gigante(paragrafo: str, max_chars: int) -> list[str]:
    """Quebra um paragrafo > max_chars em grupos de frases completas <= max_chars."""
    frases = _dividir_em_frases(paragrafo)
    grupos: list[str] = []
    atual = ""
    for frase in frases:
        if not atual or len(atual) + len(frase) + 1 <= max_chars:
            atual = (atual + " " + frase).strip() if atual else frase
        else:
            grupos.append(atual)
            atual = frase
    if atual:
        grupos.append(atual)
    return grupos


def chunk_por_fronteira(paginas: list[dict], min_chars: int = 300, max_chars: int = 1200) -> list[dict]:
    """Cria chunks por fronteira natural de paragrafos (300-1200 chars)."""
    chunks = []
    chunk_id = 0

    def emitir(texto: str, pag: dict) -> None:
        nonlocal chunk_id
        chunk_id += 1
        chunks.append(_criar_chunk(texto, pag, chunk_id))

    for pag in paginas:
        texto = _limpar(pag["texto"])
        if not texto:
            continue
        paragrafos = _dividir_em_paragrafos(texto)
        atual = ""
        for par in paragrafos:
            if len(par) > max_chars:
                # paragrafo gigante: quebra em frases completas
                if atual:
                    emitir(atual, pag)
                    atual = ""
                for grupo in _quebrar_paragrafo_gigante(par, max_chars):
                    emitir(grupo, pag)
                continue
            if not atual:
                atual = par
            elif len(atual) + len(par) + 2 <= max_chars:
                atual = atual + "\n\n" + par
            else:
                # proximo paragrafo nao cabe no chunk atual
                if len(atual) >= min_chars:
                    emitir(atual, pag)
                    atual = par
                else:
                    # chunk atual pequeno demais: mescla com o proximo (overflow controlado)
                    atual = atual + "\n\n" + par
        if atual:
            emitir(atual, pag)

    logger.info("Chunking por paragrafos completo: {} chunks (min_chars={}, max_chars={})",
                len(chunks), min_chars, max_chars)
    return chunks


# Alias para compatibilidade (testes/docs antigas).
chunk_paginas = chunk_por_fronteira


def _criar_chunk(texto: str, pag: dict, chunk_id: int) -> dict:
    return {
        "id": chunk_id,
        "doc_id": pag["doc_id"],
        "arquivo": pag["arquivo"],
        "pagina": pag["pagina"],
        "titulo": pag.get("titulo"),
        "texto": texto.strip(),
        "tokens": len(texto.split()),
        "chars": len(texto),
    }

"""Ingestao de documentos para o RAG."""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger
from pypdf import PdfReader


def extrair_texto_pdf(caminho: Path) -> list[dict]:
    """Extrai texto pagina a pagina de um PDF."""
    doc_id = caminho.stem
    paginas = []
    try:
        reader = PdfReader(str(caminho))
        for i, page in enumerate(reader.pages, start=1):
            texto = page.extract_text() or ""
            paginas.append({
                "doc_id": doc_id,
                "arquivo": caminho.name,
                "pagina": i,
                "texto": texto.strip(),
            })
        logger.debug("PDF {}: {} paginas extraidas", caminho.name, len(paginas))
    except Exception as e:
        logger.error("Erro ao extrair {}: {}", caminho, e)
    return paginas


def extrair_texto_md(caminho: Path) -> list[dict]:
    """Extrai texto de um arquivo Markdown, tratando-o como uma unica pagina."""
    doc_id = caminho.stem
    texto = caminho.read_text(encoding="utf-8", errors="ignore")
    return [{
        "doc_id": doc_id,
        "arquivo": caminho.name,
        "pagina": 1,
        "texto": texto.strip(),
    }]


def detectar_titulo(texto: str) -> str | None:
    """Heuristica simples para detectar titulo/secao: primeira linha em caixa alta."""
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    for linha in linhas[:5]:
        if linha.isupper() and len(linha) > 3:
            return linha
    return None


def ingest(diretorio: Path) -> list[dict]:
    """Ingestiona todos os PDFs e Markdowns de um diretorio."""
    paginas = []
    for caminho in sorted(diretorio.iterdir()):
        if caminho.suffix.lower() == ".pdf":
            paginas.extend(extrair_texto_pdf(caminho))
        elif caminho.suffix.lower() in (".md", ".markdown"):
            paginas.extend(extrair_texto_md(caminho))
    # adiciona titulo detectado
    for p in paginas:
        p["titulo"] = detectar_titulo(p["texto"])
    logger.info("Ingestao completa: {} paginas de {} documentos", len(paginas), diretorio)
    return paginas


def salvar_paginas(paginas: list[dict], saida: Path) -> None:
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(paginas, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Paginas salvas em {}", saida)

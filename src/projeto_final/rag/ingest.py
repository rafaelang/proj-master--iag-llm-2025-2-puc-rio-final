"""Ingestao de documentos para o RAG."""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger
from pypdf import PdfReader


# ------------------------------------------------------- limpeza de "mobilia" de slides

def _tem_email(linha: str) -> bool:
    """True se a linha contem um endereco de e-mail."""
    return re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", linha) is not None


def _tem_url(linha: str) -> bool:
    """True se a linha contem uma URL (http/https/www)."""
    return re.search(r"\b(?:https?://|www\.)", linha, re.IGNORECASE) is not None


def _eh_numerica(linha: str) -> bool:
    """True se a linha e apenas numerica (numero de pagina/slide, paginacao, percentual)."""
    return re.fullmatch(r"[\d][\d\s.,:/%\-–—]*", linha) is not None


def _eh_cabecalho_curto(linha: str) -> bool:
    """True se a linha e um cabecalho curto de slide (Prof., Professor, Pagina, Slide)."""
    if len(linha) > 60:
        return False
    return re.search(r"\bprof\.|professor|p[áa]gin|slide", linha, re.IGNORECASE) is not None


def _limpar_pagina(texto: str) -> str:
    """Remove a 'mobilia' de slides que dilui o sinal do BM25/embedding.

    Por linha: e-mails, URLs, linhas numericas, cabecalhos curtos
    (Prof., Professor, Pagina, Slide) e linhas vazias/curtas (len < 3).
    """
    linhas_limpas = []
    for linha in texto.splitlines():
        l = linha.strip()
        if not l or len(l) < 3:
            continue
        if _tem_email(l) or _tem_url(l) or _eh_numerica(l) or _eh_cabecalho_curto(l):
            continue
        linhas_limpas.append(l)
    return "\n".join(linhas_limpas)


def extrair_texto_pdf(caminho: Path) -> list[dict]:
    """Extrai texto pagina a pagina de um PDF."""
    doc_id = caminho.name  # nome do arquivo COM extensao (ex.: nlp_aula06_rag_avancado_ocr.pdf)
    paginas = []
    try:
        reader = PdfReader(str(caminho))
        for i, page in enumerate(reader.pages, start=1):
            texto = _limpar_pagina(page.extract_text() or "")
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
    doc_id = caminho.name  # nome do arquivo COM extensao
    texto = _limpar_pagina(caminho.read_text(encoding="utf-8", errors="ignore"))
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

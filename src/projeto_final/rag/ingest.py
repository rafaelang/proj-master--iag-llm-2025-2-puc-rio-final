"""Ingestao de documentos para o RAG."""

from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
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


_CARACTERES_TEXTUAIS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "áéíóúâêôãõç "
)


def _razao_alfanumerica(linha: str) -> float:
    """Fracao de caracteres textuais (letras/numeros/acentos pt-BR + espaco) na linha."""
    if not linha:
        return 1.0
    return sum(1 for c in linha if c in _CARACTERES_TEXTUAIS) / len(linha)


def _tem_caractere_controle(linha: str) -> bool:
    """True se a linha tem bytes de controle/binarios (ord < 32 exceto tab/\n/\r)."""
    return any(ord(c) < 32 and c not in "\t\n\r" for c in linha)


def _espacamento_ruim(linha: str) -> bool:
    """True se a media de caracteres por palavra indica garbage (colada ou letra a letra).

    media = len(linha) / n_palavras (split considera \n como separador).
    - media > 20: string gigante sem espacos (descartar sempre);
    - media < 3:  letras isoladas - descartar apenas com >= 5 palavras
      (evita falso-positivo em linhas curtas como 'O que e?').
    """
    palavras = linha.split()
    if not palavras:
        return True
    media = len(linha) / len(palavras)
    if media > 20:
        return True
    if media < 3 and len(palavras) >= 5:
        return True
    return False

def _limpar_pagina(texto: str) -> str:
    """Remove mobilia de slides e GARBAGE de PDF que diluem o sinal do BM25/embedding.

    Por linha, descarta:
    - e-mails, URLs, linhas numericas, cabecalhos curtos (Prof./Professor/Pagina/Slide);
    - linhas vazias/curtas (< 3 chars);
    - garbage de PDF: razao alfanumerica < 70%, caracteres de controle (binarios),
      ou espacamento medio de palavra > 20 (colada) / < 3 (letra a letra).
    """
    linhas_limpas = []
    for linha in texto.splitlines():
        l = linha.strip()
        if not l or len(l) < 3:
            continue
        if _tem_email(l) or _tem_url(l) or _eh_numerica(l) or _eh_cabecalho_curto(l):
            continue
        # --- garbage de PDF ---
        if _tem_caractere_controle(l):
            continue
        if _razao_alfanumerica(l) < 0.70:
            continue
        if _espacamento_ruim(l):
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


# ------------------------------------------------------------ IPYNB (notebook)

def _celula_ipynb_para_texto(celula: dict) -> str | None:
    """Converte uma celula de notebook (markdown/code) para texto util.

    - markdown: junta o source;
    - code: junta o source e descarta outputs (evita poluir BM25 com saidas/bases64).
    Retorna None para celulas irrelevantes (ex.: vazias).
    """
    tipo = celula.get("cell_type")
    if tipo not in ("markdown", "code"):
        return None
    source = celula.get("source", [])
    if isinstance(source, str):
        source = [source]
    texto = "".join(source)
    texto = texto.strip()
    if not texto:
        return None
    # descarta imagens/base64 embutidas no markdown ({{image}} / data:image)
    texto = re.sub(r"<img[^>]*>", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", " ", texto)
    # celulas de code por vezes sao apenas imports/bibliotecas - mantemos, pois
    # o notebook tambem ensina API (ex.: rag, llm). Nao ha cleanup agressivo.
    return texto


def extrair_texto_ipynb(caminho: Path) -> list[dict]:
    """Extrai texto de um Jupyter Notebook (JSON), tratando-o como pagina unica.

    Concatena celulas markdown + code (sem outputs). Celulas sao separadas por
    marcadores discretos para o chunking respeitar a fronteira natural.
    """
    doc_id = caminho.name
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("Erro ao extrair {}: {}", caminho, e)
        return []
    if not isinstance(dados, dict):
        logger.error("Estrutura de notebook inesperada em {}", caminho)
        return []
    partes: list[str] = []
    for celula in dados.get("cells", []):
        linhas = _celula_ipynb_para_texto(celula)
        if linhas:
            partes.append(f"# celula {len(partes) + 1}\n{linhas}")
    texto = _limpar_pagina("\n\n".join(partes))
    return [{
        "doc_id": doc_id,
        "arquivo": caminho.name,
        "pagina": 1,
        "texto": texto.strip(),
    }]


# ------------------------------------------------------------ PPTX (slides)


def _slides_pptx_para_texto(caminho: Path) -> list[str]:
    """Extrai o texto de cada slide de um .pptx (zip) via stdlib.

    Retorna lista de blocos de texto, um por slide (ordem de ppt/slides/slideN.xml).
    """
    blocos: list[tuple[int, str]] = []
    try:
        with zipfile.ZipFile(caminho) as z:
            nomes_slides = sorted(
                (n for n in z.namelist()
                 if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                key=lambda n: int(re.search(r"\d+", n.split("/")[-1]).group()),
            )
            if not nomes_slides:
                return []
            for nome in nomes_slides:
                n = int(re.search(r"\d+", nome.split("/")[-1]).group())
                xml = z.read(nome)
                raiz = ET.fromstring(xml)
                partes: list[str] = []
                for no in raiz.iter():
                    if no.tag.endswith("}t") and no.text:
                        t = no.text.strip()
                        if len(t) >= 2:
                            partes.append(t)
                blocos.append((n, "\n".join(partes)))
    except (zipfile.BadZipFile, ET.ParseError, KeyError, ValueError) as e:
        logger.error("Erro ao extrair {}: {}", caminho, e)
        return []
    blocos.sort(key=lambda b: b[0])
    return [b for _, b in blocos]


def extrair_texto_pptx(caminho: Path) -> list[dict]:
    """Extrai texto de uma apresentacao PPTX, pagina por slide."""
    doc_id = caminho.name
    paginas = []
    for i, texto in enumerate(_slides_pptx_para_texto(caminho), start=1):
        texto = _limpar_pagina(texto)
        if texto.strip():
            paginas.append({
                "doc_id": doc_id,
                "arquivo": caminho.name,
                "pagina": i,
                "texto": texto.strip(),
            })
    if not paginas:
        logger.warning("PPTX sem texto extraido: {}", caminho)
    return paginas


def detectar_titulo(texto: str) -> str | None:
    """Heuristica simples para detectar titulo/secao: primeira linha em caixa alta."""
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    for linha in linhas[:5]:
        if linha.isupper() and len(linha) > 3:
            return linha
    return None


def ingest(diretorio: Path) -> list[dict]:
    """Ingestiona todos os documentos suportados de um diretorio.

    Formatos: PDF, Markdown, Jupyter Notebook (.ipynb) e PowerPoint (.pptx).
    """
    paginas = []
    for caminho in sorted(diretorio.iterdir()):
        if caminho.suffix.lower() == ".pdf":
            paginas.extend(extrair_texto_pdf(caminho))
        elif caminho.suffix.lower() in (".md", ".markdown"):
            paginas.extend(extrair_texto_md(caminho))
        elif caminho.suffix.lower() == ".ipynb":
            paginas.extend(extrair_texto_ipynb(caminho))
        elif caminho.suffix.lower() == ".pptx":
            paginas.extend(extrair_texto_pptx(caminho))
    # adiciona titulo detectado
    for p in paginas:
        p["titulo"] = detectar_titulo(p["texto"])
    logger.info("Ingestao completa: {} paginas de {} documentos", len(paginas), diretorio)
    return paginas


def salvar_paginas(paginas: list[dict], saida: Path) -> None:
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(paginas, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Paginas salvas em {}", saida)

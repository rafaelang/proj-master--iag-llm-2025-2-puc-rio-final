"""Extracao de imagens do corpus + OCR local (RapidOCR) — v0.3.

Fluxo (tudo cacheado em `data/processed/rag_v3/`, fora do Git):
  1) `registro_imagens()`  -> varre os PDFs de data/raw com PyMuPDF, deduplica por
     sha1 dos pixels e grava `imagens_registro.json`;
  2) `ocr_imagens()`       -> roda o RapidOCR (ONNX, CPU, local) em cada imagem
     unica ainda nao processada e grava `imagens_ocr.json`;
  3) `chunks_de_imagem()`  -> limpa o texto OCR, mede a NOVIDADE (tokens que o
     layer de texto da pagina nao tem) e grava `imagens_chunks.json`.

Conceito da v0.3: a "visao corrige o texto" quando uma figura (diagrama, tabela,
slide rasterizado) contem informacao que o layer de texto do PDF nao expoe. Os
chunks de imagem entram no mesmo indice RAG (texto + imagem), permitindo
recuperar e citar o conteudo visual.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pymupdf
from loguru import logger

from projeto_final import config
from projeto_final.bm25 import normalizar

# Limiar de novidade: fracao dos tokens do OCR que NAO existe no texto da pagina.
# Acima disso a imagem e tratada como figura com conteudo proprio (vira chunk).
NOVO_RATIO_MIN = 0.45
# Sinal de dominio: fracao dos tokens do OCR que existem no texto do DOCUMENTO
# inteiro. Filtra screenshots/UI garbage (menus, toolbars) que so poluem o indice.
SINAL_DOMINIO_MIN = 0.40
# Comprimento minimo (chars) do texto OCR limpo para virar chunk.
OCR_CHUNK_MIN_CHARS = 30
# Enriquecimento: prefixo com o cabecalho/titulo do slide (layer de texto) para
# ancorar o chunk da figura semanticamente — sem isso, listas de rotulos OCR
# nao competem com os chunks de prosa no ranking hibrido.
PREFIXO_MAX = 160


def _sha1_pixmap(pix: pymupdf.Pixmap) -> str:
    """sha1 dos pixels (amostras brutas) de um Pixmap."""
    return hashlib.sha1(pix.samples).hexdigest()


def _converter_rgb(pix: pymupdf.Pixmap) -> pymupdf.Pixmap:
    """Converte pixmap com alpha/CMYK/etc. para RGB (3 canais) — seguro p/ OCR."""
    if pix.n - pix.alpha < 4:
        return pix
    return pymupdf.Pixmap(pymupdf.csRGB, pix)


def registro_imagens(force: bool = False) -> list[dict]:
    """Extrai imagens raster de todos os PDFs de data/raw e grava o registro.

    - Deduplica por sha1 dos pixels (mesma figura em varias paginas/PDFs);
    - ignora imagens com lado < IMG_MIN_LADO (icones/logos/bullets);
    - para cada imagem unica guarda a 1a ocorrencia (doc_id, pagina, indice).
    """
    if not force and config.IMG_REGISTRO_PATH.exists():
        logger.debug("Registro de imagens carregado de {}", config.IMG_REGISTRO_PATH)
        return json.loads(config.IMG_REGISTRO_PATH.read_text(encoding="utf-8"))

    ocorrencias: dict[str, dict] = {}
    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    for caminho in pdfs:
        doc = pymupdf.open(str(caminho))
        try:
            for pno in range(len(doc)):
                page = doc[pno]
                for indice, info in enumerate(page.get_images(full=True)):
                    xref = info[0]
                    try:
                        pix = _converter_rgb(pymupdf.Pixmap(doc, xref))
                    except Exception:
                        continue
                    if pix.width < config.IMG_MIN_LADO or pix.height < config.IMG_MIN_LADO:
                        continue
                    sha = _sha1_pixmap(pix)
                    item = ocorrencias.setdefault(
                        sha,
                        {"sha1": sha, "w": pix.width, "h": pix.height, "ocorrencias": []},
                    )
                    item["ocorrencias"].append(
                        {"doc_id": caminho.name, "pagina": pno + 1, "indice": indice}
                    )
        finally:
            doc.close()

    registro = []
    for item in ocorrencias.values():
        # ocorrencias iguais repetidas (mesma pagina/indice) nao contam 2x
        unicas = sorted(
            {(o["doc_id"], o["pagina"], o["indice"]) for o in item["ocorrencias"]}
        )
        principal = unicas[0]
        item["ocorrencias"] = [
            {"doc_id": d, "pagina": p, "indice": i} for d, p, i in unicas
        ]
        item["principal"] = {"doc_id": principal[0], "pagina": principal[1], "indice": principal[2]}
        registro.append(item)

    registro.sort(key=lambda r: (-len(r["ocorrencias"]), r["principal"]["doc_id"], r["principal"]["pagina"]))
    config.RAG_V3_DIR.mkdir(parents=True, exist_ok=True)
    config.IMG_REGISTRO_PATH.write_text(
        json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Registro de imagens: {} imagens unicas (de {} PDFs)", len(registro), len(pdfs))
    return registro

def _extrair_pixmap(item: dict) -> tuple[pymupdf.Pixmap, Path]:
    """Re-extrai o Pixmap da imagem a partir da ocorrencia principal do item."""
    p = item["principal"]
    caminho = config.RAW_DIR / p["doc_id"]
    doc = pymupdf.open(str(caminho))
    try:
        page = doc[p["pagina"] - 1]
        imagens = page.get_images(full=True)
        xref = imagens[p["indice"]][0]
        pix = _converter_rgb(pymupdf.Pixmap(doc, xref))
    finally:
        doc.close()
    return pix, caminho


def _limpar_linhas_ocr(linhas: list[tuple[str, float]]) -> str:
    """Limpa o texto OCR linha a linha (mobilia/garbage de figura)."""
    vistos: set[str] = set()
    saida: list[str] = []
    for txt, conf in linhas:
        l = txt.strip()
        if not l or len(l) < 1:
            continue
        if any(ord(c) < 32 and c not in "\t" for c in l):
            continue
        # linha sem nenhuma letra/número relevante (so simbolos) — descarta
        if not re.search(r"[a-zA-Z0-9áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ]", l):
            continue
        chave = l.lower()
        if chave in vistos:
            continue
        vistos.add(chave)
        saida.append(l)
    return "\n".join(saida)


def _ordenar_linhas(resultado: list[list]) -> list[tuple[str, float]]:
    """Ordena linhas do RapidOCR de cima p/ baixo, da esquerda p/ direita."""
    linhas = []
    for box, txt, conf in resultado:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        linhas.append((min(ys), min(xs), str(txt), float(conf)))
    linhas.sort(key=lambda t: (t[0] // 8, t[1]))
    return [(t[2], t[3]) for t in linhas]


def ocr_imagens(force: bool = False) -> dict:
    """Roda RapidOCR (ONNX, CPU) nas imagens unicas ainda nao processadas.

    Resultado cacheado em `imagens_ocr.json` (imagens processadas nao repetem).
    """
    registro = registro_imagens(force=False)
    cache: dict = {}
    if not force and config.IMG_OCR_PATH.exists():
        cache = json.loads(config.IMG_OCR_PATH.read_text(encoding="utf-8"))

    pendentes = [r for r in registro if r["sha1"] not in cache]
    if not pendentes:
        logger.info("OCR: {} imagens ja processadas (nada a fazer)", len(cache))
        return cache

    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    t0 = time.time()
    for i, item in enumerate(pendentes, 1):
        pix, _ = _extrair_pixmap(item)
        png = pix.tobytes("png")
        try:
            res, _el = ocr(png)
        except Exception as e:  # imagem corrompida/exotica — registra vazio
            logger.warning("OCR falhou (sha {}): {}", item["sha1"][:10], e)
            res = None
        linhas = _ordenar_linhas(res) if res else []
        confs = [c for _, c in linhas] or [0.0]
        cache[item["sha1"]] = {
            "linhas": [t for t, _ in linhas],
            "confs": confs,
            "texto_bruto": "\n".join(t for t, _ in linhas),
            "conf_media": round(sum(confs) / len(confs), 3),
        }
        if i % 25 == 0 or i == len(pendentes):
            logger.info("OCR progresso {}/{}", i, len(pendentes))
        t0 = time.time()

    config.RAG_V3_DIR.mkdir(parents=True, exist_ok=True)
    config.IMG_OCR_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("OCR concluido: {} imagens processadas", len(cache))
    return cache



def _garantir_paginas_texto() -> None:
    """Garante que data/processed/rag/paginas.json existe (layer de texto do corpus)."""
    caminho = config.RAG_DIR / "paginas.json"
    if caminho.exists():
        return
    from projeto_final.rag.ingest import ingest, salvar_paginas
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    paginas = ingest(config.RAW_DIR)
    salvar_paginas(paginas, caminho)


def _texto_da_pagina(doc_id: str, pagina: int) -> str:
    """Texto limpo do layer de texto da pagina (usado para medir novidade)."""
    _garantir_paginas_texto()
    caminho = config.RAG_DIR / "paginas.json"
    if not caminho.exists():
        return ""
    paginas = json.loads(caminho.read_text(encoding="utf-8"))
    for p in paginas:
        if p["doc_id"] == doc_id and p["pagina"] == pagina:
            return p.get("texto", "")
    return ""


def _medir_novidade(texto_ocr: str, texto_pagina: str) -> float:
    """Fracao dos tokens do OCR que nao existem no layer de texto da pagina."""
    tokens_ocr = set(normalizar(texto_ocr))
    if not tokens_ocr:
        return 0.0
    tokens_pag = set(normalizar(texto_pagina))
    novos = tokens_ocr - tokens_pag
    return round(len(novos) / len(tokens_ocr), 3)


def _tokens_doc(doc_id: str) -> set[str]:
    """Tokens normalizados do DOCUMENTO inteiro (layer de texto) — cache em memoria."""
    if not hasattr(_tokens_doc, "_cache"):
        _tokens_doc._cache = {}
        _garantir_paginas_texto()
        caminho = config.RAG_DIR / "paginas.json"
        if caminho.exists():
            paginas = json.loads(caminho.read_text(encoding="utf-8"))
            por_doc: dict[str, list[str]] = {}
            for p in paginas:
                por_doc.setdefault(p["doc_id"], []).append(p.get("texto", ""))
            _tokens_doc._cache = {
                d: set(normalizar("\n".join(txt))) for d, txt in por_doc.items()
            }
    return _tokens_doc._cache.get(doc_id, set())


def _medir_sinal_dominio(texto_ocr: str, doc_id: str) -> float:
    """Fracao dos tokens do OCR que aparecem no documento inteiro (sinal de dominio)."""
    tokens_ocr = set(normalizar(texto_ocr))
    if not tokens_ocr:
        return 0.0
    doc = _tokens_doc(doc_id)
    return round(len(tokens_ocr & doc) / len(tokens_ocr), 3)


def _prefixo_pagina(texto_pag: str) -> str:
    """Cabecalho curto do slide (primeiras linhas nao-vazias) para ancorar a figura."""
    linhas = [l.strip() for l in texto_pag.splitlines() if l.strip()]
    prefixo = ""
    for l in linhas:
        if prefixo and len(prefixo) + len(l) + 1 > PREFIXO_MAX:
            break
        prefixo = (prefixo + " " + l).strip() if prefixo else l
        if len(prefixo) >= PREFIXO_MAX:
            break
    return prefixo[:PREFIXO_MAX]


def _criar_chunk_imagem(
    item: dict, ocr_item: dict, texto: str, prefixo: str, novo_ratio: float, sinal: float, cid: int
) -> dict:
    p = item["principal"]
    texto_final = texto if not prefixo else f"{prefixo} | FIGURA: {texto}"
    return {
        "id": cid,
        "doc_id": p["doc_id"],
        "arquivo": p["doc_id"],
        "pagina": p["pagina"],
        "titulo": prefixo or None,
        "texto": texto_final,
        "tokens": len(texto_final.split()),
        "chars": len(texto_final),
        "tipo": "imagem",
        "imagem": {
            "sha1": item["sha1"],
            "indice": p["indice"],
            "w": item["w"],
            "h": item["h"],
        },
        "ocr_conf_media": ocr_item["conf_media"],
        "novo_ratio": novo_ratio,
        "sinal_dominio": sinal,
    }


def chunks_de_imagem(force: bool = False) -> list[dict]:
    """Constroi chunks de imagem (figuras com conteudo novo p/ o corpus).

    Filtros: texto OCR limpo >= OCR_CHUNK_MIN_CHARS chars e novidade >=
    NOVO_RATIO_MIN (a figura precisa acrescentar algo que o texto nao tem).
    """
    if not force and config.IMG_CHUNKS_PATH.exists():
        logger.debug("Chunks de imagem carregados de {}", config.IMG_CHUNKS_PATH)
        return json.loads(config.IMG_CHUNKS_PATH.read_text(encoding="utf-8"))

    registro = registro_imagens()
    ocr_cache = ocr_imagens()
    chunks = []
    descartados = {"curtos": 0, "sem_novidade": 0, "sem_sinal": 0, "vazios": 0}
    for item in registro:
        ocr_item = ocr_cache.get(item["sha1"])
        if not ocr_item or not ocr_item["linhas"]:
            descartados["vazios"] += 1
            continue
        texto = _limpar_linhas_ocr(list(zip(ocr_item["linhas"], ocr_item["confs"])))
        if len(texto) < OCR_CHUNK_MIN_CHARS:
            descartados["curtos"] += 1
            continue
        p = item["principal"]
        texto_pag = _texto_da_pagina(p["doc_id"], p["pagina"])
        novo_ratio = _medir_novidade(texto, texto_pag)
        if novo_ratio < NOVO_RATIO_MIN:
            descartados["sem_novidade"] += 1
            continue
        sinal = _medir_sinal_dominio(texto, p["doc_id"])
        if sinal < SINAL_DOMINIO_MIN:
            descartados["sem_sinal"] += 1
            continue
        prefixo = _prefixo_pagina(texto_pag)
        chunks.append(_criar_chunk_imagem(item, ocr_item, texto, prefixo, novo_ratio, sinal, len(chunks) + 1))

    config.RAG_V3_DIR.mkdir(parents=True, exist_ok=True)
    config.IMG_CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Chunks de imagem: {} criados (descartados: {})",
        len(chunks), descartados,
    )
    return chunks


def analisar_imagem_bytes(dados: bytes) -> dict:
    """OCR local (RapidOCR) de uma imagem enviada pelo usuario (bytes).

    Usado pelo endpoint `POST /rag/imagem/analisar` — mesma etapa de "visao"
    que indexa as figuras do corpus, agora sobre imagem avulsa.
    """
    import cv2
    from rapidocr_onnxruntime import RapidOCR

    arr = cv2.imdecode(np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("nao foi possivel decodificar a imagem")
    ocr = RapidOCR()
    res, _el = ocr(arr)
    linhas = _ordenar_linhas(res) if res else []
    confs = [c for _, c in linhas] or [0.0]
    texto = _limpar_linhas_ocr(linhas)
    return {
        "linhas": [t for t, _ in linhas],
        "texto": texto,
        "chars": len(texto),
        "conf_media": round(sum(confs) / len(confs), 3),
    }

"""Pipeline RAG: pergunta -> chunks (texto [+ imagem v0.3]) -> resposta com citacao."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from projeto_final import config
from projeto_final.rag.chunk import chunk_por_fronteira
from projeto_final.rag.ingest import ingest, salvar_paginas
from projeto_final.rag.retrieve import recuperar
from projeto_final import llm as llm_mod


def carregar_chunks() -> list[dict]:
    """Carrega chunks de TEXTO do disco (v0.2); constroi do corpus se nao existir."""
    if config.RAG_CHUNK_PATH.exists():
        logger.debug("Carregando chunks de {}", config.RAG_CHUNK_PATH)
        return json.loads(config.RAG_CHUNK_PATH.read_text(encoding="utf-8"))

    logger.info("Chunks nao encontrados. Ingestionando corpus...")
    paginas = ingest(config.RAW_DIR)
    salvar_paginas(paginas, config.RAG_DIR / "paginas.json")
    chunks = chunk_por_fronteira(paginas)
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    config.RAG_CHUNK_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("{} chunks salvos em {}", len(chunks), config.RAG_CHUNK_PATH)
    return chunks


# ------------------------------------------------------------- v0.3 - imagem

def carregar_corpus_v3() -> list[dict]:
    """Corpus COMBINADO da v0.3: chunks de texto (v0.2) + chunks de imagem (OCR).

    Se o corpus v3 ja foi persistido em RAG_V3_DIR, apenas carrega; caso contrario
    monta (texto + imagem), renumera os ids e persiste.
    """
    if config.RAG_V3_CHUNK_PATH.exists():
        logger.debug("Corpus v3 carregado de {}", config.RAG_V3_CHUNK_PATH)
        return json.loads(config.RAG_V3_CHUNK_PATH.read_text(encoding="utf-8"))

    from projeto_final.rag.imagem import chunks_de_imagem

    texto = carregar_chunks()
    imagem = chunks_de_imagem()
    combinado = []
    n = 1
    for c in texto + imagem:
        combinado.append({**c, "id": n, "tipo": c.get("tipo", "texto")})
        n += 1
    config.RAG_V3_DIR.mkdir(parents=True, exist_ok=True)
    config.RAG_V3_CHUNK_PATH.write_text(
        json.dumps(combinado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Corpus v3 construido: {} chunks ({} texto + {} imagem)",
        len(combinado), len(texto), len(imagem),
    )
    return combinado


# ------------------------------------------------------------- resposta

def _formatar_contexto(chunks: list[dict]) -> str:
    linhas = []
    for i, chunk in enumerate(chunks, start=1):
        origem = chunk.get("tipo", "texto")
        sufixo = " (imagem/figura OCR)" if origem == "imagem" else ""
        linhas.append(f"[{i}] {chunk['doc_id']}, pagina {chunk['pagina']}{sufixo}: {chunk['texto']}")
    return "\n\n".join(linhas)


def responder(pergunta: str, top_k: int = 5) -> dict:
    """Executa o pipeline RAG completo (v0.2: somente texto) e retorna dict."""
    return _responder(pergunta, top_k=top_k, base=None, usar_imagens=False)


def responder_v3(pergunta: str, top_k: int = 5) -> dict:
    """Pipeline RAG da v0.3: corpus texto + imagem (OCR de figuras)."""
    return _responder(pergunta, top_k=top_k, base=config.RAG_V3_DIR, usar_imagens=True)


# ------------------------------------------------------------- referencias

def _formatar_referencias(resposta: str, chunks: list[dict]) -> dict:
    """Renumera os marcadores [N] da resposta e monta o rodape de referencias.

    Abordagem deterministica (anti-alucinacao):
    - remove rodape que o modelo por ventura tenha escrito (linhas "[N] arquivo.pdf");
    - mantem apenas marcadores validos (indices do contexto, 1..len(chunks));
    - renumera em [1], [2], ... na ordem de primeira aparicao;
    - monta o rodape "[n] doc_id" e uma variante limpa para o TTS (sem rodape e
      sem marcadores).
    """
    import re as _re

    corpo = "\n".join(
        linha for linha in resposta.splitlines()
        if not _re.match(r"^\[\d+\]\s+[A-Za-z0-9_\-.]*\.pdf\b", linha.strip())
    )
    validos = {i + 1: chunks[i] for i in range(len(chunks))}
    ordem: list[tuple[int, dict]] = []
    vistos: set[int] = set()

    def _sub(m: _re.Match) -> str:
        idx = int(m.group(1))
        if idx not in validos:
            return ""  # marcador fora do contexto (alucinado): remove
        if idx not in vistos:
            vistos.add(idx)
            ordem.append((idx, validos[idx]))
        return f"[{len(ordem)}]"

    texto_marcado = _re.sub(r"\[(\d+)\]", _sub, corpo)
    texto_marcado = _re.sub(r"[ \t]{2,}", " ", texto_marcado).strip()
    rodape = "\n".join(f"[{i}] {c['doc_id']}" for i, (_idx, c) in enumerate(ordem, 1))
    resposta_final = f"{texto_marcado}\n\n{rodape}" if rodape else texto_marcado
    resposta_tts = _re.sub(r"\[\d+\]", " ", texto_marcado).strip()
    referencias = [
        {"n": i, "doc_id": c["doc_id"], "pagina": c.get("pagina")}
        for i, (_idx, c) in enumerate(ordem, 1)
    ]
    return {
        "resposta": resposta_final,
        "resposta_tts": resposta_tts,
        "referencias": referencias,
        "texto_marcado": texto_marcado,
    }


def _responder(pergunta: str, top_k: int, base: Path | None, usar_imagens: bool) -> dict:
    import time

    t0 = time.time()
    if usar_imagens:
        chunks = carregar_corpus_v3()
    else:
        chunks = carregar_chunks()
    recuperados = recuperar(pergunta, chunks, top_k=top_k, base=base, rerank=config.RAG_RERANK)
    contexto = _formatar_contexto(recuperados)
    sistema = config.ler_prompt("v0.2/rag_sistema.txt")
    resposta, meta_llm = llm_mod.responder_com_contexto(pergunta, contexto, sistema=sistema)
    refs = _formatar_referencias(resposta, recuperados)
    abstencao = llm_mod.detectar_abstencao(resposta)
    logger.info("RAG: pergunta='{}' abstencao={} chunks={} imagens={}", pergunta, abstencao, len(recuperados), usar_imagens)
    return {
        "pergunta": pergunta,
        "resposta": refs["resposta"],          # texto + rodapé de referências
        "resposta_tts": refs["resposta_tts"],  # só o conteúdo (para o TTS)
        "referencias": refs["referencias"],
        "abstencao": abstencao,
        "chunks": recuperados,
        "contexto": contexto,
        "latencia_s": round(time.time() - t0, 2),
        "modelo_llm": meta_llm.get("modelo"),
        "uso": meta_llm.get("uso"),
        "usar_imagens": usar_imagens,
    }


def reconstruir_indices() -> None:
    """Forca a reconstrucao dos indices de texto a partir do corpus."""
    from projeto_final.rag.index import construir_indices
    chunks = carregar_chunks()
    construir_indices(chunks, force=True)


def reconstruir_indices_v3() -> None:
    """Forca a reconstrucao do indice combinado texto + imagem (v0.3)."""
    from projeto_final.rag.index import construir_indices
    chunks = carregar_corpus_v3()
    construir_indices(chunks, force=True, base=config.RAG_V3_DIR)

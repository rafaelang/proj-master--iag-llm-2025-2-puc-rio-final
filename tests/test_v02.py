"""Testes da v0.2 — RAG."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from projeto_final import config
from projeto_final.main import app
from projeto_final.rag.chunk import chunk_por_fronteira
from projeto_final.rag.ingest import ingest
from projeto_final.rag.index import construir_indices
from projeto_final.rag.retrieve import recuperar


def test_ingest_nao_vazio():
    paginas = ingest(config.RAW_DIR)
    assert len(paginas) > 0
    assert all("doc_id" in p and "texto" in p for p in paginas)


def test_chunk_fronteira_natural():
    paginas = ingest(config.RAW_DIR)
    chunks = chunk_por_fronteira(paginas)
    assert len(chunks) > 0
    assert all("id" in c and "doc_id" in c and "texto" in c for c in chunks)


def test_indices_constroem():
    paginas = ingest(config.RAW_DIR)
    chunks = chunk_por_fronteira(paginas)
    bm25, embeddings, ids = construir_indices(chunks, force=True)
    assert embeddings.shape[0] == len(chunks)
    assert len(ids) == len(chunks)


def test_recupera_chunks():
    paginas = ingest(config.RAW_DIR)
    chunks = chunk_por_fronteira(paginas)
    # rerank=False: teste basico do RRF, sem baixar o cross-encoder (~1.1 GB)
    top = recuperar("RAG retrieval augmented generation", chunks, top_k=5, rerank=False)
    assert len(top) == 5
    assert all("doc_id" in c for c in top)


def test_rag_saude():
    client = TestClient(app)
    res = client.get("/rag/saude")
    assert res.status_code == 200
    data = res.json()
    assert "chunks_indexados" in data


def test_rag_perguntar():
    client = TestClient(app)
    res = client.post("/rag/perguntar?pergunta=O+que+e+RAG?")
    assert res.status_code == 200
    data = res.json()
    assert "resposta" in data
    assert "abstencao" in data
    assert "chunks" in data


def test_abstencao_negativa():
    client = TestClient(app)
    res = client.post("/rag/perguntar?pergunta=Quem+foi+o+primeiro+presidente+do+Brasil?")
    assert res.status_code == 200
    data = res.json()
    assert data["abstencao"] is True


def test_prompt_rag_acentuacao_e_abstencao_estrita():
    """O prompt de sistema deve estar acentuado e exigir NAO_SEI isolado."""
    from projeto_final import llm

    sistema = config.ler_prompt("v0.2/rag_sistema.txt")
    assert sistema
    assert "não" in sistema and "informação" in sistema and "NAO_SEI" in sistema
    assert "Voce" not in sistema and "Nao" not in sistema  # sem versoes sem acento
    assert "SOMENTE com o token exato NAO_SEI" in sistema

    tts_prompt = llm.PROMPT_TTS
    assert "não" in tts_prompt and "acentuação" in tts_prompt


def test_detectar_abstencao_variantes():
    from projeto_final import llm

    for r in (
        "NAO_SEI",
        "NÃO_SEI",
        "Não sei",
        "não sei",
        "O contexto não cobre isso. NAO_SEI",
        "não há informações suficientes",
        "não tenho informações sobre isso",
    ):
        assert llm.detectar_abstencao(r) is True, r
    for r in ("O RAG une recuperação e geração.", "", "RAG é a sigla de Retrieval-Augmented Generation."):
        assert llm.detectar_abstencao(r) is False, r

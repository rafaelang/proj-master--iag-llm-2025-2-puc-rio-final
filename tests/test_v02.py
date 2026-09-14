"""Testes da v0.2 — RAG."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from projeto_final import config
from projeto_final.main import app
from projeto_final.rag import pipeline as rag_pipeline
from projeto_final.rag.chunk import chunk_por_fronteira
from projeto_final.rag.ingest import ingest
from projeto_final.rag.index import construir_indices
from projeto_final.rag.retrieve import recuperar


TEXTO_FIXTURE = (
    "O RAG une a recuperacao de trechos relevantes do corpus com a geracao "
    "aumentada por contexto. O indice combina BM25 (lexico) com embeddings "
    "densos (semantico) e funde os dois rankings por RRF. O sistema responde "
    "com citacao [n] da fonte e abstem-se quando a pergunta esta fora do "
    "corpus. Esta e a base dos testes de fronteira natural de chunking." * 2
)


def test_ingest_nao_vazio(tmp_path):
    """Contrato de ingest no formato minimo (fixture rapida; o corpus completo
    nao e re-ingerido nesta suite)."""
    (tmp_path / "a.md").write_text(TEXTO_FIXTURE, encoding="utf-8")
    paginas = ingest(tmp_path)
    assert len(paginas) > 0
    assert all("doc_id" in p and "texto" in p for p in paginas)


def test_chunk_fronteira_natural(tmp_path):
    (tmp_path / "a.md").write_text(TEXTO_FIXTURE, encoding="utf-8")
    paginas = ingest(tmp_path)
    chunks = chunk_por_fronteira(paginas)
    assert len(chunks) > 0
    assert all("id" in c and "doc_id" in c and "texto" in c for c in chunks)


def test_indices_constroem(tmp_path):
    """Construcao BM25+embeddings em base propria (tmp) — nunca rebuilda o
    indice de producao; acrescenta sanidade de leitura do indice oficial."""
    chunks = [
        {"id": i, "doc_id": f"d{i % 3}.pdf", "pagina": (i % 5) + 1,
         "texto": "RAG une recuperacao e geracao aumentada por contexto. " * 4}
        for i in range(15)
    ]
    base = tmp_path / "idx"
    bm25, embeddings, ids = construir_indices(chunks, force=True, base=base)
    assert embeddings.shape[0] == len(chunks)
    assert len(ids) == len(chunks)
    # persistencia: leitura volta identica
    from projeto_final.rag.index import carregar_indices
    _, emb2, ids2 = carregar_indices(base)
    assert list(ids2) == ids
    # sanidade (somente leitura) do indice de producao, se existir
    if config.RAG_BM25_PATH.exists() and config.RAG_EMBEDDINGS_PATH.exists():
        from projeto_final.rag.pipeline import carregar_chunks
        _, emb_prod, ids_prod = carregar_indices(config.RAG_DIR)
        assert emb_prod.shape[0] == len(ids_prod) == len(carregar_chunks())


def test_recuperar_dedup_chunks():
    """Chunks duplicados/sobrepostos da mesma pagina nao entram 2x no resultado."""
    from projeto_final.rag.retrieve import _deduplicar

    texto = "O RAG une a recuperação de trechos com a geração de respostas."
    quase_dup = texto + " (mesma ideia repetida no mesmo slide)"
    chunks = [
        {"id": 1, "doc_id": "a.pdf", "pagina": 1, "texto": texto},
        {"id": 2, "doc_id": "a.pdf", "pagina": 1, "texto": texto},          # exata
        {"id": 3, "doc_id": "a.pdf", "pagina": 1, "texto": quase_dup},      # quase-dup
        {"id": 4, "doc_id": "a.pdf", "pagina": 2, "texto": "Conteúdo distinto da pagina seguinte."},
        {"id": 5, "doc_id": "b.pdf", "pagina": 1, "texto": texto},          # mesmo texto, outro doc: mantem
    ]
    saida = _deduplicar(chunks)
    assert [c["id"] for c in saida] == [1, 4, 5]


def test_formatar_referencias_renumera_e_monta_rodape():
    """Marcadores viram [1],[2],... na ordem de aparicao + rodape no final."""
    from projeto_final.rag.pipeline import _formatar_referencias

    chunks = [
        {"id": 10, "doc_id": "nlp_aula06.pdf", "pagina": 5},
        {"id": 20, "doc_id": "pai_aula07.pdf", "pagina": 9},
        {"id": 30, "doc_id": "tdp_aula03.pdf", "pagina": 2},
    ]
    resposta = "O RAG enriquece o contexto [3]. Ele pode usar FastAPI [1]. (invalido [9])"
    out = _formatar_referencias(resposta, chunks)

    # renumeração pela ordem de primeira aparição: [3]->[1], [1]->[2]
    assert "O RAG enriquece o contexto [1]" in out["resposta"]
    assert "usar FastAPI [2]" in out["resposta"]
    assert "[9]" not in out["resposta"]
    # rodapé sequencial com os doc_ids corretos
    assert "tdp_aula03.pdf" in out["resposta"] and "nlp_aula06.pdf" in out["resposta"]
    assert "[1] tdp_aula03.pdf" in out["resposta"]
    assert "[2] nlp_aula06.pdf" in out["resposta"]
    # variante TTS sem marcadores/rodapé
    assert "[" not in out["resposta_tts"]
    assert out["referencias"] == [
        {"n": 1, "doc_id": "tdp_aula03.pdf", "pagina": 2},
        {"n": 2, "doc_id": "nlp_aula06.pdf", "pagina": 5},
    ]


def test_recupera_chunks():
    """Recuperacao RRF usando o corpus/indice oficiais persistidos (sem
    re-ingest e sem rebuild — rapido e nao-destrutivo)."""
    import pytest
    if not (config.RAG_BM25_PATH.exists() and config.RAG_EMBEDDINGS_PATH.exists()
            and config.RAG_CHUNK_PATH.exists()):
        pytest.skip("indice/corpus de producao ausentes (rode a ingestao/construcao)")
    from projeto_final.rag.pipeline import carregar_chunks
    chunks = carregar_chunks()
    # rerank=False: teste basico do RRF, sem baixar o cross-encoder (~1.1 GB)
    top = recuperar("RAG retrieval augmented generation", chunks, top_k=5, rerank=False)
    assert len(top) == 5
    assert all("doc_id" in c for c in top)


def test_saude_rag():
    """GET /saude expoe o estado do RAG (chunks_indexados) dentro de rag."""
    client = TestClient(app)
    res = client.get("/saude")
    assert res.status_code == 200
    data = res.json()
    assert "chunks_indexados" in data["rag"]


def test_rag_responder_pipeline():
    """Pipeline RAG (chamada direta): contrato resposta/abstencao/chunks.

    v0.4: o endpoint /rag/perguntar foi removido — o RAG roda em /chat e
    /chat/imagem; o contrato continua coberto por esta chamada de pipeline.
    """
    data = rag_pipeline.responder("O que e RAG?")
    assert "resposta" in data
    assert "abstencao" in data
    assert "chunks" in data


def test_abstencao_negativa():
    """Fora do corpus -> abstencao True (mesma heuristica da v0.2)."""
    data = rag_pipeline.responder("Quem foi o primeiro presidente do Brasil?")
    assert data["abstencao"] is True


def test_rag_perguntar_endpoint_removido():
    """v0.4: /rag/perguntar e /rag/imagem/analisar foram removidos (404)."""
    client = TestClient(app)
    assert client.post("/rag/perguntar?pergunta=O+que+e+RAG?").status_code == 404
    assert client.post("/rag/imagem/analisar").status_code == 404


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

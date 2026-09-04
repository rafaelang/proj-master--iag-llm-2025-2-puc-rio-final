"""Testes do rate limiting da API (janela deslizante em memoria)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from projeto_final import config
from projeto_final.main import _grupo_e_limite, _rate_limpar, _rate_permitir, app


def test_grupo_e_limite():
    """Pagina tem limite alto; endpoints de RAG/chat (LLM) tem 4/min."""
    assert _grupo_e_limite("/") == ("geral", config.RATELIMIT_GERAL_QTD)
    assert _grupo_e_limite("/saude") == ("geral", config.RATELIMIT_GERAL_QTD)
    for path in ("/rag/saude", "/chat", "/chat/imagem"):
        assert _grupo_e_limite(path) == ("rag", config.RATELIMIT_RAG_QTD)
    assert config.RATELIMIT_RAG_QTD == 4  # decisao: 4 por minuto


def test_rate_permitir_janela():
    _rate_limpar()
    for _ in range(4):
        ok, _ = _rate_permitir("rag", "ip-1", 4, 60)
        assert ok is True
    ok, espera = _rate_permitir("rag", "ip-1", 4, 60)
    assert ok is False and espera > 0
    assert _rate_permitir("rag", "ip-2", 4, 60)[0] is True  # outro cliente segue livre
    _rate_limpar()


def test_429_no_endpoint_rag():
    config.RATELIMIT_HABILITADO = True
    config.RATELIMIT_RAG_QTD = 2
    _rate_limpar()
    try:
        client = TestClient(app)
        cab = {"X-Forwarded-For": "203.0.113.7"}
        codigos = [client.get("/rag/saude", headers=cab).status_code for _ in range(3)]
        assert codigos == [200, 200, 429]
        resp = client.get("/rag/saude", headers=cab)
        assert int(resp.headers["Retry-After"]) >= 1
        assert "429" in resp.text or "Muitas requisicoes" in resp.text
    finally:
        _rate_limpar()
        config.RATELIMIT_HABILITADO = False

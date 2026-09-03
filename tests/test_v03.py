"""Testes da v0.3 — Imagem (OCR local de figuras integrado ao RAG)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from projeto_final import config
from projeto_final.main import app


def _test_img_bytes() -> bytes:
    """PNG branco 200x100 (imagem valida, sem texto)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_golden_set_imagem_valido():
    """O dataset unico da v0.3 existe e cada pergunta aponta para uma figura."""
    assert config.RAG_IMG_GOLDEN_SET.exists()
    data = json.loads(config.RAG_IMG_GOLDEN_SET.read_text(encoding="utf-8"))
    perguntas = data["perguntas"]
    assert len(perguntas) >= 1
    for q in perguntas:
        assert q["doc_esperado"].endswith(".pdf")
        assert q["pagina_figura"] >= 1
        assert len(q["termos_esperados"]) >= 1
        assert q["deve_abster"] is False


def test_limpar_linhas_ocr_descarta_garbage():
    """Limpeza de OCR remove linhas vazias/simbolos e duplicatas consecutivas."""
    from projeto_final.rag.imagem import _limpar_linhas_ocr

    linhas = [("  nome ", 0.99), ("---", 0.9), ("nome", 0.95), ("CPF", 1.0), ("", 0.0)]
    saida = _limpar_linhas_ocr(linhas)
    assert saida == "nome\nCPF"


def test_analisar_imagem_endpoint():
    """POST /rag/imagem/analisar retorna o contrato (OCR) para uma imagem."""
    client = TestClient(app)
    res = client.post(
        "/rag/imagem/analisar",
        files={"file": ("branco.png", _test_img_bytes(), "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert set(data) == {"linhas", "texto", "chars", "conf_media"}


def test_analisar_imagem_vazia_erro():
    client = TestClient(app)
    res = client.post("/rag/imagem/analisar", files={"file": ("vazio.png", b"", "image/png")})
    assert res.status_code == 400


@pytest.mark.skipif(
    not config.RAG_V3_CHUNK_PATH.exists(),
    reason="corpus v0.3 (texto+imagem) ainda nao processado — rode scripts/avaliadores/avaliar_v3.py",
)
def test_corpus_v3_mescla_texto_e_imagem():
    """O corpus v0.3 soma chunks de texto (v0.2) e de imagem (OCR) com ids unicos."""
    from projeto_final.rag.pipeline import carregar_chunks, carregar_corpus_v3

    texto = carregar_chunks()
    v3 = carregar_corpus_v3()
    n_img = sum(1 for c in v3 if c.get("tipo") == "imagem")
    assert len(v3) == len(texto) + n_img
    assert n_img >= 1
    ids = [c["id"] for c in v3]
    assert len(ids) == len(set(ids))  # ids unicos e sequenciais
    assert ids == list(range(1, len(v3) + 1))

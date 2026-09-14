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


def test_analisar_imagem_bytes():
    """OCR local (bytes) — contrato {linhas,texto,chars,conf_media}.

    v0.4: o endpoint /rag/imagem/analisar foi removido; a funcao
    analisar_imagem_bytes (usada pela pipeline v0.3) continua coberta direto.
    """
    from projeto_final.rag.imagem import analisar_imagem_bytes

    data = analisar_imagem_bytes(_test_img_bytes())
    assert set(data) == {"linhas", "texto", "chars", "conf_media"}


def test_analisar_imagem_bytes_vazia_erro():
    import cv2

    from projeto_final.rag.imagem import analisar_imagem_bytes

    with pytest.raises((ValueError, cv2.error)):
        analisar_imagem_bytes(b"")


def test_config_modelo_visao_definido():
    """A constante do modelo de visao (deepseek-vision) existe em config."""
    from projeto_final import config as cfg

    assert cfg.DEEPSEEK_VISION_MODEL
    assert cfg.DEEPSEEK_VISION_MAX_TOKENS >= 100


def test_chat_extensao_invalida():
    """POST /chat com arquivo de extensao desconhecida -> 415."""
    client = TestClient(app)
    res = client.post("/chat", files={"file": ("nota.txt", b"texto", "text/plain")})
    assert res.status_code == 415


def test_chat_arquivo_vazio():
    """POST /chat com arquivo vazio (PNG 0 bytes) -> 400."""
    client = TestClient(app)
    res = client.post("/chat", files={"file": ("vazia.png", b"", "image/png")})
    assert res.status_code == 400


def test_chat_imagem_fluxo_json():
    """Imagem no /chat unificado: visao -> prompt RAG -> mesmo fluxo de resposta
    -> JSON unificado com texto + audio_base64 (contrato v0.4)."""
    from unittest.mock import patch

    fake = {
        "resposta": "A imagem mostra um diagrama. [1]",
        "audio": b"RIFFwav", "abstencao": False,
        "referencias": [{"n": 1, "doc_id": "nlp_aula06_rag_avancado_ocr.pdf", "pagina": 5}],
        "latencia_s": {"llm": 0.4, "tts": 0.3},
    }
    with patch(
        "projeto_final.main.llm.descrever_imagem",
        return_value=("grafico de barras", {"modelo": "fake-vision", "latencia_s": 0.2}),
    ) as vis, patch("projeto_final.main._pipeline_resposta", return_value=fake) as pl:
        client = TestClient(app)
        res = client.post("/chat", files={"file": ("fig.png", _test_img_bytes(), "image/png")})
    assert res.status_code == 200
    corpo = res.json()
    assert corpo["tipo_entrada"] == "imagem"
    assert corpo["texto"] == "grafico de barras"
    assert corpo["resposta"].startswith("A imagem mostra")
    assert corpo["audio_base64"]             # texto + audio na saida
    assert corpo["modelo_entrada"] == "fake-vision"
    assert corpo["latencia"]["entrada"] == 0.2
    vis.assert_called_once()
    pl.assert_called_once()
    pergunta = pl.call_args.args[0]
    assert "grafico de barras" in pergunta and "Fale sobre o assunto" in pergunta
    assert pl.call_args.kwargs["com_audio"] is True


@pytest.mark.skipif(
    not config.DEEPSEEK_API_KEY or not config.DEEPSEEK_VISION_MODEL,
    reason="requer DEEPSEEK_API_KEY e modelo de visao configurados",
)
def test_descrever_imagem_integracao():
    """Smoke de integracao: modelo de visao descreve uma imagem de teste."""
    from projeto_final import llm

    descricao, meta = llm.descrever_imagem(_test_img_bytes(), "image/png")
    assert isinstance(descricao, str) and descricao.strip()
    assert meta["modelo"] == config.DEEPSEEK_VISION_MODEL



@pytest.mark.skipif(
    not config.RAG_V3_CHUNK_PATH.exists(),
    reason="corpus v0.3 (texto+imagem) ainda nao processado — rode scripts/avaliadores/avaliar_v3.py",
)
def test_corpus_v3_mescla_texto_e_imagem():
        """O merge texto+imagem da v0.3 foi absorvido pelo pipeline unico atual.

        A v0.3 combinava chunks de texto (v0.2) e de imagem (OCR) num corpus
        proprio; a partir da v0.4+ o pipeline consolidou tudo em `carregar_chunks`
        (chunks.json oficial). Este teste garante (a) o legado v3 continua
        persistido e carregavel com os campos de chunk, e (b) o corpus atual
        (>= v3) e idempotente na leitura.
        """
        from projeto_final.rag.pipeline import carregar_chunks, carregar_corpus_v3

        v3 = carregar_corpus_v3()
        assert len(v3) > 0
        assert all("doc_id" in c and "pagina" in c and "texto" in c for c in v3)
        atuais = carregar_chunks()
        assert len(atuais) >= len(v3)  # o pipeline atual consolidou o merge v0.3
        assert all("id" in c and "doc_id" in c and "texto" in c for c in atuais)

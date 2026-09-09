"""Testes da v0.1."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from projeto_final.main import app
from projeto_final.wer import normalizar, wer


def test_normalizar():
    assert normalizar("RAG, embeddings!") == ["rag", "embeddings"]


def test_normalizar_texto_tts():
    """A camada de normalizacao p/ voz remove marcadores/rodape e nao truncada."""
    from projeto_final.tts import _normalizar_texto_tts

    # resposta completa (marcadores inline + rodapé) -> le apenas o conteudo
    resposta = (
        "O RAG enriquece o contexto [1] e pode usar FastAPI [2].\n"
        "\n"
        "[1] nlp_aula06_rag_avancado_ocr.pdf\n"
        "[2] pai_aula07_deploy_fastapi.pdf"
    )
    saida = _normalizar_texto_tts(resposta)
    assert "nlp_aula06" not in saida and "pai_aula07" not in saida
    assert "O RAG enriquece o contexto e pode usar FastAPI." in saida  # texto completo
    # abstenção pura vira frase natural
    assert "Não sei responder" in _normalizar_texto_tts("NAO_SEI")
    assert "Não sei responder" in _normalizar_texto_tts("Não sei")
    # markdown/símbolos removidos, espaços colapsados
    saida2 = _normalizar_texto_tts("  **RAG**  é *recuperação*  ")
    assert saida2 == "RAG é recuperação"


def test_tts_vozes_alternativas():
    """As vozes pt-BR alternativas ficam disponiveis via VOZES (padrao: cadu)."""
    from projeto_final import config
    from projeto_final.tts import VOZES, VOZ_PADRAO

    assert VOZ_PADRAO == "pt_BR-cadu-medium"
    assert config.PIPER_VOICE in VOZES
    for alternativa in ("pt_BR-cadu-medium", "pt_BR-jeff-medium", "pt_BR-edresson-low"):
        assert alternativa in VOZES
        assert VOZES[alternativa]["base"].startswith("https://huggingface.co/rhasspy/piper-voices")


def test_config_prosodia_piper():
    """Envs de prosodia (length_scale etc.) existem e sao positivas."""
    from projeto_final import config

    assert config.PIPER_LENGTH_SCALE > 0.0
    assert config.PIPER_NOISE_SCALE > 0.0
    assert config.PIPER_NOISE_W_SCALE > 0.0
    assert config.PIPER_VOLUME > 0.0


def test_wer_perfeito():
    assert wer("o que e rag", "o que e rag") == 0.0


def test_wer_50():
    assert wer("a b c", "a x c") == 1 / 3


def test_root_serve_index():
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "Chat por voz" in res.text


def test_saude():
    """GET /saude (unico endpoint de saude) expoe asr/llm/tts/rag."""
    client = TestClient(app)
    res = client.get("/saude")
    assert res.status_code == 200
    data = res.json()
    assert data["asr"]["backend"] == "faster-whisper (CPU)"
    assert {"asr", "llm", "tts", "rag"} <= set(data)


def test_saude_subrotas_removidas():
    """Apenas /saude existe — /voz/saude e /rag/saude retornam 404."""
    client = TestClient(app)
    for path in ("/voz/saude", "/rag/saude"):
        assert client.get(path).status_code == 404


def test_chat_com_audio_silencioso():
    """WAV mudo para /chat -> 422 (transcricao vazia) ou 200 com JSON
    {texto, resposta, audio_base64} (saida texto + audio)."""
    from projeto_final.tts import _wav_mudo

    wav = _wav_mudo(duracao_s=1.0)
    client = TestClient(app)
    res = client.post(
        "/chat",
        files={"file": ("teste.wav", io.BytesIO(wav), "audio/wav")},
    )
    assert res.status_code in (200, 422)
    if res.status_code == 200:
        corpo = res.json()
        assert corpo["tipo_entrada"] == "audio"
        assert corpo["resposta"]
        assert corpo["audio_base64"]

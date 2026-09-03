"""Testes da v0.1."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from projeto_final.main import app
from projeto_final.wer import normalizar, wer


def test_normalizar():
    assert normalizar("RAG, embeddings!") == ["rag", "embeddings"]


def test_normalizar_texto_tts():
    """A camada de normalizacao p/ voz remove citacoes/ruido e expande NAO_SEI."""
    from projeto_final.tts import _normalizar_texto_tts

    # citação RAG é removida para leitura
    saida = _normalizar_texto_tts("RAG une recuperacao e geracao. [1] nlp_aula06_rag_avancado_ocr.pdf")
    assert "nlp_aula06" not in saida and "[1]" not in saida
    # abstenção pura vira frase natural
    assert "Não sei responder" in _normalizar_texto_tts("NAO_SEI")
    assert "Não sei responder" in _normalizar_texto_tts("Não sei")
    # markdown/símbolos removidos, espaços colapsados
    saida2 = _normalizar_texto_tts("  **RAG**  é *recuperação*  ")
    assert saida2 == "RAG é recuperação"


def test_tts_vozes_alternativas():
    """As vozes pt-BR alternativas ficam disponiveis via VOZES."""
    from projeto_final import config
    from projeto_final.tts import VOZES, VOZ_PADRAO

    assert VOZ_PADRAO == "pt_BR-faber-medium"
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
    client = TestClient(app)
    res = client.get("/voz/saude")
    assert res.status_code == 200
    data = res.json()
    assert data["asr"]["backend"] == "faster-whisper (CPU)"


def test_chat_com_audio_silencioso():
    """Envia um WAV mudo para /chat e verifica resposta com headers."""
    from projeto_final.tts import _wav_mudo

    wav = _wav_mudo(duracao_s=1.0)
    client = TestClient(app)
    res = client.post(
        "/chat",
        files={"file": ("teste.wav", io.BytesIO(wav), "audio/wav")},
    )
    assert res.status_code in (200, 422)
    if res.status_code == 200:
        assert "X-Transcription" in res.headers
        assert "X-Answer" in res.headers
        assert res.headers["content-type"] == "audio/wav"

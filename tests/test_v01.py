"""Testes da v0.1."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from projeto_final.main import app
from projeto_final.wer import normalizar, wer


def test_normalizar():
    assert normalizar("RAG, embeddings!") == ["rag", "embeddings"]


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

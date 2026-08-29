"""TTS local com Piper — v0.1."""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path

from loguru import logger

from projeto_final import config

VOZES = {
    "pt_BR-faber-medium": {
        "base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/pt/pt_BR/faber/medium/pt_BR-faber-medium",
        "exts": (".onnx", ".onnx.json"),
    },
}
VOZ_PADRAO = "pt_BR-faber-medium"
SAMPLE_RATE = 22050

_voice_cache: dict = {}


def _caminho_modelo(voz: str) -> Path:
    modelo_env = config.PIPER_MODEL
    if modelo_env:
        p = Path(modelo_env)
        if not p.exists():
            raise FileNotFoundError(f"PIPER_MODEL nao encontrado: {p}")
        return p

    if voz not in VOZES:
        raise ValueError(f"Voz desconhecida: {voz!r}")

    info = VOZES[voz]
    config.PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx = config.PIPER_MODELS_DIR / f"{voz}.onnx"
    if not onnx.exists():
        import urllib.request
        for ext in info["exts"]:
            destino = config.PIPER_MODELS_DIR / f"{voz}{ext}"
            if destino.exists():
                continue
            url = info["base"] + ext
            logger.info("Baixando voz Piper {}: {}", voz, url)
            urllib.request.urlretrieve(url, destino)
    return onnx


def _get_voz(voz: str):
    if voz not in _voice_cache:
        from piper import PiperVoice
        modelo = _caminho_modelo(voz)
        logger.debug("Carregando voz Piper: {}", voz)
        _voice_cache[voz] = PiperVoice.load(str(modelo))
    return _voice_cache[voz]


def _chunks_para_wav(chunks, sample_rate: int | None = None) -> bytes:
    buf = io.BytesIO()
    sr = sample_rate
    primeiro = True
    with wave.open(buf, "wb") as w:
        for chunk in chunks:
            if primeiro:
                w.setnchannels(chunk.sample_channels)
                w.setsampwidth(chunk.sample_width)
                w.setframerate(sr if sr is not None else chunk.sample_rate)
                primeiro = False
            w.writeframes(chunk.audio_int16_array.tobytes())
    return buf.getvalue()


def _wav_mudo(duracao_s: float = 0.5, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(sample_rate * duracao_s))
    return buf.getvalue()


def sintetizar(texto: str, voz: str | None = None, sample_rate: int | None = None) -> tuple[bytes, float]:
    """Sintetiza texto em WAV. Retorna (bytes_wav, latencia_s)."""
    import time
    t0 = time.time()
    backend = config.TTS_BACKEND.lower()
    if backend == "echo":
        logger.debug("TTS backend=echo — devolvendo silencio")
        return _wav_mudo(), time.time() - t0
    if backend != "piper":
        raise ValueError(f"TTS_BACKEND desconhecido: {backend!r}")

    voz = voz or config.PIPER_VOICE
    voice = _get_voz(voz)
    wav = _chunks_para_wav(voice.synthesize(texto), sample_rate)
    latencia = time.time() - t0
    logger.debug("TTS finalizado: {} bytes em {} s", len(wav), round(latencia, 2))
    return wav, latencia

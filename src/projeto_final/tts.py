"""TTS local com Piper — v0.1 (+ normalizacao de texto e prosodia configuravel).

Melhorias:
- Vozes pt-BR alternativas cadastradas (faber/cadu/jeff medium, edresson low);
- Camada de normalizacao p/ leitura em voz alta (remove citacoes [N], URLs,
  simbolos de marcacao e expande o token de abstencao NAO_SEI);
- SynthesisConfig aplicado (length_scale>1.0 desacelera e reduz consoantes
  "atropeladas"; noise_scale/noise_w_scale/volume configuraveis por env).
"""

from __future__ import annotations

import io
import os
import re
import unicodedata
import wave
from pathlib import Path

from loguru import logger

from projeto_final import config


def _base_url_voz(nome: str, qualidade: str) -> str:
    """URL base de download de uma voz pt-BR no rhasspy/piper-voices (v1.0.0)."""
    return (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        f"pt/pt_BR/{nome}/{qualidade}/pt_BR-{nome}-{qualidade}"
    )


VOZES = {
    "pt_BR-faber-medium": {"base": _base_url_voz("faber", "medium"), "exts": (".onnx", ".onnx.json")},
    "pt_BR-cadu-medium": {"base": _base_url_voz("cadu", "medium"), "exts": (".onnx", ".onnx.json")},
    "pt_BR-jeff-medium": {"base": _base_url_voz("jeff", "medium"), "exts": (".onnx", ".onnx.json")},
    "pt_BR-edresson-low": {"base": _base_url_voz("edresson", "low"), "exts": (".onnx", ".onnx.json")},
}
VOZ_PADRAO = "pt_BR-faber-medium"
SAMPLE_RATE = 22050

_voice_cache: dict = {}


# ------------------------------------------------------- normalizacao p/ leitura

def _sem_acentos(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _normalizar_texto_tts(texto: str) -> str:
    """Prepara o texto para leitura em voz alta pelo Piper.

    - token de abstencao puro (NAO_SEI / NÃO_SEI / "nao sei") vira frase falada;
    - remove citacoes do RAG no formato "[N] doc_id[, pagina X]" (nao legiveis);
    - remove URLs/e-mails e simbolos de marcacao (markdown, listas, negrito);
    - colapsa espacos em branco.
    """
    if not texto:
        return ""
    t = texto.strip()

    # abstenção pura -> frase natural para a voz
    sem = _sem_acentos(t).replace("_", " ").replace("-", " ")
    sem = re.sub(r"\s+", " ", sem)
    if re.fullmatch(r"nao sei(?:[.!]?)", sem):
        return "Não sei responder com base nos materiais do curso."

    # citacoes RAG: "[1] arquivo.pdf, pagina 5" (ate o fim da linha ou prox. [N])
    t = re.sub(r"\s*\[\d+\][^\n\[]*", " ", t)
    # URLs e e-mails
    t = re.sub(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " ", t)
    # marcacao residual: * _ # ` > | e barras de separacao de rotulo OCR
    t = re.sub(r"[\u0000-\u001f*_#`>|]", " ", t)
    # colapsa espacos e sinais de pontuacao repetidos
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ------------------------------------------------------- carregamento da voz

def _caminho_modelo(voz: str) -> Path:
    modelo_env = config.PIPER_MODEL
    if modelo_env:
        p = Path(modelo_env)
        if not p.exists():
            raise FileNotFoundError(f"PIPER_MODEL nao encontrado: {p}")
        return p

    if voz not in VOZES:
        raise ValueError(f"Voz desconhecida: {voz!r} (disponiveis: {', '.join(sorted(VOZES))})")

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


def _config_sintese():
    """SynthesisConfig do Piper a partir das envs do config."""
    from piper.config import SynthesisConfig

    return SynthesisConfig(
        length_scale=config.PIPER_LENGTH_SCALE,
        noise_scale=config.PIPER_NOISE_SCALE,
        noise_w_scale=config.PIPER_NOISE_W_SCALE,
        volume=config.PIPER_VOLUME,
    )


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
    texto_limpo = _normalizar_texto_tts(texto)
    if not texto_limpo:
        logger.warning("Texto vazio apos normalizacao; devolvendo silencio")
        return _wav_mudo(), time.time() - t0

    voice = _get_voz(voz)
    syn = _config_sintese()
    logger.debug(
        "TTS: voz={} len_scale={} noise={} noise_w={} volume={} texto={} chars",
        voz, syn.length_scale, syn.noise_scale, syn.noise_w_scale, syn.volume, len(texto_limpo),
    )
    wav = _chunks_para_wav(voice.synthesize(texto_limpo, syn_config=syn), sample_rate)
    latencia = time.time() - t0
    logger.debug("TTS finalizado: {} bytes em {} s", len(wav), round(latencia, 2))
    return wav, latencia

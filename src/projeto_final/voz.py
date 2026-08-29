"""Pipeline de voz — v0.1."""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

from loguru import logger

from projeto_final import config
from projeto_final.wer import wer

AUDIO_EXTS = ("wav", "mp3", "m4a", "ogg", "flac", "webm")

_modelo_cache: dict = {}


def _get_modelo(modelo: str) -> "WhisperModel":
    if modelo not in _modelo_cache:
        from faster_whisper import WhisperModel
        config.WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("Carregando modelo faster-whisper: {}", modelo)
        _modelo_cache[modelo] = WhisperModel(
            modelo,
            device="cpu",
            compute_type="int8",
            download_root=str(config.WHISPER_CACHE_DIR),
        )
    return _modelo_cache[modelo]


def transcrever(audio: str, prompt: str | None = None, modelo: str = "small") -> tuple[str, float]:
    """Transcreve um arquivo localmente com faster-whisper (pt)."""
    t0 = time.time()
    logger.debug("Iniciando transcrição de {}", audio)
    model = _get_modelo(modelo)
    kwargs = {
        "language": "pt",
        "vad_filter": True,
    }
    if prompt:
        kwargs["initial_prompt"] = prompt
        logger.debug("Usando prompt de vocabulario de dominio")
    segments, _info = model.transcribe(audio, **kwargs)
    texto = " ".join(seg.text.strip() for seg in segments).strip()
    latencia = time.time() - t0
    logger.debug("Transcrição finalizada: {} ({} s)", texto, round(latencia, 2))
    return texto, latencia


def ler_referencias(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["frases"]


def encontrar_audio(diretorio: str, frase_id: int) -> str | None:
    for ext in AUDIO_EXTS:
        for pattern in (
            f"{diretorio}/{frase_id:02d}.{ext}",
            f"{diretorio}/{frase_id}.{ext}",
        ):
            hits = glob.glob(pattern)
            if hits:
                return hits[0]
    return None


def avaliar(refs_path: str, audios_dir: str, prompt: str | None = None, modelo: str = "small") -> dict:
    refs = ler_referencias(refs_path)
    resultados = []

    for frase in refs:
        fid, ref = frase["id"], frase["referencia"]
        audio = encontrar_audio(audios_dir, fid)
        if not audio:
            logger.warning("frase {}: audio nao encontrado em {}", fid, audios_dir)
            continue

        hip_sem, lat_sem = transcrever(audio, prompt=None, modelo=modelo)
        hip_com, lat_com = transcrever(audio, prompt=prompt, modelo=modelo)

        resultados.append({
            "id": fid,
            "referencia": ref,
            "audio": audio,
            "sem_vocab": {"texto": hip_sem, "wer": round(wer(ref, hip_sem), 4), "latencia_s": round(lat_sem, 2)},
            "com_vocab": {"texto": hip_com, "wer": round(wer(ref, hip_com), 4), "latencia_s": round(lat_com, 2)},
        })
        logger.debug("frase {}: WER sem={}, com={}", fid, resultados[-1]["sem_vocab"]["wer"], resultados[-1]["com_vocab"]["wer"])

    if not resultados:
        raise SystemExit("Nenhum audio encontrado")

    n = len(resultados)
    med_sem = sum(r["sem_vocab"]["wer"] for r in resultados) / n
    med_com = sum(r["com_vocab"]["wer"] for r in resultados) / n

    logger.info("WER medio: sem={:.4f}, com={:.4f}, ganho={:+.4f}", med_sem, med_com, med_sem - med_com)

    return {
        "resumo": {
            "n_frases": n,
            "modelo": modelo,
            "wer_medio_sem_vocab": round(med_sem, 4),
            "wer_medio_com_vocab": round(med_com, 4),
            "ganho_absoluto": round(med_sem - med_com, 4),
            "backend": "faster-whisper local (CPU)",
        },
        "frases": resultados,
    }

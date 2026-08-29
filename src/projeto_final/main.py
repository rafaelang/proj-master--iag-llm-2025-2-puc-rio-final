"""FastAPI — v0.1."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from loguru import logger
from starlette.concurrency import run_in_threadpool
from urllib.parse import quote

from projeto_final import config, llm, tts, voz

RAIZ = config.RAIZ
STATIC_INDEX = RAIZ / "static" / "index.html"
CHAT_EXTS = (*voz.AUDIO_EXTS,)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Assistente Master IAG & LLM - API de Voz",
        description="Recebe audio, transcreve, responde via LLM e devolve audio sintetizado.",
        version="0.1.0",
    )

    @app.on_event("startup")
    def startup():
        logger.info("Iniciando API v0.1 em {}", RAIZ)
        if not config.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY nao configurada — endpoint /chat usara TTS echo se LLM falhar")

    @app.get("/", response_class=HTMLResponse)
    def pagina() -> HTMLResponse:
        if not STATIC_INDEX.exists():
            raise HTTPException(status_code=404, detail="index.html nao encontrado")
        return HTMLResponse(STATIC_INDEX.read_text(encoding="utf-8"))

    @app.get("/voz/saude")
    def saude() -> dict:
        return {
            "asr": {
                "backend": "faster-whisper (CPU)",
                "modelo": config.ASR_MODELO,
                "vocabulario_dominio": config.PROMPT_VOCABULARIO.exists(),
            },
            "llm": {
                "provedor": "deepseek",
                "configurado": bool(config.DEEPSEEK_API_KEY),
            },
            "tts": {
                "backend": config.TTS_BACKEND,
                "voz": config.PIPER_VOICE,
            },
        }

    @app.post("/chat")
    async def chat(file: UploadFile = File(...)) -> Response:
        logger.info("Recebendo audio: {}", file.filename)
        ext = (file.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in CHAT_EXTS:
            logger.warning("Extensao nao suportada: {}", ext)
            raise HTTPException(
                status_code=415,
                detail=f"Extensao nao suportada: .{ext} (aceitas: {', '.join(CHAT_EXTS)})",
            )

        dados = await file.read()
        if not dados:
            raise HTTPException(status_code=400, detail="audio vazio")

        t_total = time.time()
        p = await run_in_threadpool(_pipeline_voz, dados, ext)
        logger.info("Pipeline finalizado em {} s", round(time.time() - t_total, 2))

        return Response(
            content=p["audio"],
            media_type="audio/wav",
            headers={
                "X-Transcription": quote(p["transcricao"]),
                "X-Answer": quote(p["resposta"]),
            },
        )

    return app


def _pipeline_voz(dados: bytes, ext: str) -> dict:
    """ASR -> LLM -> TTS em thread separada."""
    prompt = config.ler_prompt_vocabulario()

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(dados)
        path = tmp.name
    try:
        texto, lat_asr = voz.transcrever(path, prompt=prompt, modelo=config.ASR_MODELO)
    finally:
        os.unlink(path)

    if not texto:
        logger.warning("Transcricao vazia")
        raise HTTPException(status_code=422, detail="Nao foi possivel transcrever o audio")

    try:
        resposta, meta_llm = llm.responder(texto)
    except Exception as e:
        logger.error("LLM falhou: {}. Usando resposta local.", e)
        resposta = "Desculpe, nao consegui consultar o modelo agora. Tente novamente."
        meta_llm = {"modelo": config.DEEPSEEK_MODEL, "uso": None, "latencia_s": 0.0}

    audio, lat_tts = tts.sintetizar(resposta)

    return {
        "transcricao": texto,
        "resposta": resposta,
        "audio": audio,
        "latencia_s": {
            "asr": round(lat_asr, 2),
            "llm": meta_llm["latencia_s"],
            "tts": round(lat_tts, 2),
        },
    }


app = create_app()

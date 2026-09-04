"""FastAPI — v0.3 (RAG texto+imagem + voz + visao multimodal)."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from loguru import logger
from starlette.concurrency import run_in_threadpool
from urllib.parse import quote

from projeto_final import config, llm, tts, voz
from projeto_final.rag import pipeline as rag_pipeline

RAIZ = config.RAIZ
STATIC_INDEX = RAIZ / "static" / "index.html"
CHAT_EXTS = (*voz.AUDIO_EXTS,)
# v0.3 - imagens no chat (visao multimodal -> descricao -> RAG -> TTS)
IMAGEM_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tif", "tiff"}

MIME_POR_EXT = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
    "tif": "image/tiff", "tiff": "image/tiff",
}

# ------------------------------------------------------- rate limiting (in-memory)
_janelas: dict[tuple[str, str], deque] = {}
_janelas_lock = threading.Lock()


def _rate_permitir(grupo: str, chave: str, qtd: int, periodo: int) -> tuple[bool, float]:
    """Janela deslizante por (grupo, chave): permite <= qtd a cada periodo s.

    Retorna (permitido, espera_ate_proxima_s)."""
    agora = time.monotonic()
    k = (grupo, chave)
    with _janelas_lock:
        fila = _janelas.setdefault(k, deque())
        while fila and agora - fila[0] >= periodo:
            fila.popleft()
        if len(fila) < qtd:
            fila.append(agora)
            return True, 0.0
        espera = periodo - (agora - fila[0])
        return False, max(espera, 0.0)


def _rate_limpar() -> None:
    """Limpa o estado do rate limit (usado em testes)."""
    with _janelas_lock:
        _janelas.clear()


def _grupo_e_limite(path: str) -> tuple[str, int]:
    """Endpoint generativo (LLM/custo): /rag/* e /chat* -> 4/min. Pagina e
    demais rotas -> limite alto (padrao 300/min)."""
    if path.startswith("/rag/") or path.startswith("/chat"):
        return "rag", config.RATELIMIT_RAG_QTD
    return "geral", config.RATELIMIT_GERAL_QTD


def _ip_cliente(request: Request) -> str:
    """IP real do cliente: prioriza X-Forwarded-For (proxy do HF/nginx)."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "desconhecido"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Assistente Master IAG & LLM - API de Voz, Imagem e RAG",
        description="Recebe audio ou imagem, transcreve/descreve, responde via RAG e devolve audio sintetizado.",
        version="0.3.0",
    )

    @app.on_event("startup")
    def startup():
        logger.info("Iniciando API v0.3 em {}", RAIZ)
        if not config.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY nao configurada — endpoint /chat usara TTS echo se LLM falhar")
        try:
            rag_pipeline.carregar_chunks()
            logger.info("Chunks RAG (texto) carregados com sucesso")
        except Exception as e:
            logger.warning("RAG ainda nao configurado: {}", e)
        # v0.3 - imagens (OCR): ativa o corpus combinado se ja processado
        if config.RAG_V3_CHUNK_PATH.exists():
            try:
                chunks_v3 = rag_pipeline.carregar_corpus_v3()
                logger.info("Corpus v0.3 carregado: {} chunks (texto + imagem)", len(chunks_v3))
            except Exception as e:
                logger.warning("Corpus v0.3 indisponivel: {}", e)

    @app.middleware("http")
    async def _throttling(request: Request, call_next):
        """Rate limit por IP: '/rag/*' e '/chat*' (geram LLM) -> 4/min;
        '/' (pagina) e demais rotas -> limite alto (configuravel)."""
        if not config.RATELIMIT_HABILITADO:
            return await call_next(request)
        grupo, qtd = _grupo_e_limite(request.url.path)
        ip = _ip_cliente(request)
        permitido, espera = _rate_permitir(grupo, ip, qtd, config.RATELIMIT_PERIODO_S)
        if not permitido:
            retry = max(1, int(espera) + 1)
            logger.warning("Rate limit {}: {} excedido por {}", grupo, qtd, ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Muitas requisicoes. Aguarde e tente novamente."},
                headers={"Retry-After": str(retry)},
            )
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def pagina() -> HTMLResponse:
        if not STATIC_INDEX.exists():
            raise HTTPException(status_code=404, detail="index.html nao encontrado")
        return HTMLResponse(STATIC_INDEX.read_text(encoding="utf-8"))

    @app.get("/saude")
    def saude() -> dict:
        chunks = []
        try:
            chunks = rag_pipeline.carregar_chunks()
        except Exception:
            pass
        n_img = None
        corpus_v3 = None
        try:
            if config.RAG_V3_CHUNK_PATH.exists():
                corpus_v3 = rag_pipeline.carregar_corpus_v3()
                n_img = sum(1 for c in corpus_v3 if c.get("tipo") == "imagem")
        except Exception:
            pass
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
            "rag": {
                "chunks_indexados": len(chunks),
                "corpus_v3": {
                    "ativo": corpus_v3 is not None,
                    "chunks_total": len(corpus_v3) if corpus_v3 else 0,
                    "chunks_imagem": n_img if n_img is not None else 0,
                },
                "corpus_raw": list(config.RAW_DIR.glob("*.*")),
            },
        }

    @app.get("/voz/saude")
    def saude_voz() -> dict:
        return saude()

    @app.get("/rag/saude")
    def saude_rag() -> dict:
        return saude()["rag"]

    @app.post("/rag/perguntar")
    async def rag_perguntar(pergunta: str, imagens: bool = False,
                            agentes: bool = False) -> dict:
        logger.info("RAG /rag/perguntar: {} (imagens={}, agentes={})",
                    pergunta, imagens, agentes)
        try:
            if agentes:
                # v0.4 - fluxo multiagente (roteador + geradores com fallback);
                # sempre usa o corpus texto+imagem (v0.3).
                from projeto_final import agentes as agentes_mod

                return await run_in_threadpool(agentes_mod.responder_agentes, pergunta)
            if imagens:
                return await run_in_threadpool(rag_pipeline.responder_v3, pergunta)
            return await run_in_threadpool(rag_pipeline.responder, pergunta)
        except Exception as e:
            logger.error("Erro no RAG: {}", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/rag/imagem/analisar")
    async def rag_imagem_analisar(file: UploadFile = File(...)) -> dict:
        """Analisa uma imagem enviada (OCR local RapidOCR) e retorna o texto.

        A v0.3 integra a visao ao RAG: o OCR de figuras do corpus entra no indice
        (`/rag/perguntar?imagens=true`). Este endpoint permite analisar imagens
        avulsas do usuario com a mesma etapa de visao.
        """
        from projeto_final.rag.imagem import analisar_imagem_bytes

        logger.info("RAG /rag/imagem/analisar: {}", file.filename)
        dados = await file.read()
        if not dados:
            raise HTTPException(status_code=400, detail="imagem vazia")
        try:
            return await run_in_threadpool(analisar_imagem_bytes, dados)
        except Exception as e:
            logger.error("Erro ao analisar imagem: {}", e)
            raise HTTPException(status_code=500, detail=str(e))

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

    @app.post("/chat/imagem")
    async def chat_imagem(file: UploadFile = File(...)) -> Response:
        """Chat por imagem: visao multimodal -> descricao -> RAG -> TTS.

        Mesmo contrato do /chat (audio/wav + X-Transcription/X-Answer):
        X-Transcription carrega a descricao gerada pelo modelo de visao, que e
        a pergunta enviada ao RAG.
        """
        logger.info("Recebendo imagem: {}", file.filename)
        ext = (file.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in IMAGEM_EXTS:
            logger.warning("Extensao de imagem nao suportada: {}", ext)
            raise HTTPException(
                status_code=415,
                detail=f"Extensao nao suportada: .{ext} (aceitas: {', '.join(sorted(IMAGEM_EXTS))})",
            )

        dados = await file.read()
        if not dados:
            raise HTTPException(status_code=400, detail="imagem vazia")

        t_total = time.time()
        try:
            p = await run_in_threadpool(_pipeline_imagem, dados, ext)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Erro no pipeline de imagem: {}", e)
            raise HTTPException(status_code=502, detail=f"Falha ao processar a imagem: {e}")
        logger.info(
            "Pipeline de imagem finalizado em {} s (visao={})",
            round(time.time() - t_total, 2), p.get("modelo_visao"),
        )

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
    """ASR -> texto -> (RAG -> TTS). Em thread separada."""
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

    return _pipeline_resposta(texto, lat_asr=lat_asr)


def _pipeline_imagem(dados: bytes, ext: str) -> dict:
    """Visao multimodal (deepseek-vision) -> prompt RAG -> TTS.

    A visao retorna ASSUNTO/TERMOS/SINTESE da imagem; o RAG recebe um PROMPT
    ("fale sobre o assunto retratado") — nao a descricao crua como pergunta.
    A resposta final enviada ao usuario e a do RAG (texto + audio).
    """
    t_visao = time.time()
    mime = MIME_POR_EXT.get(ext, "image/png")
    conteudo, meta_visao = llm.descrever_imagem(dados, mime)
    if not conteudo:
        logger.warning("Modelo de visao nao retornou conteudo")
        raise HTTPException(status_code=502, detail="O modelo de visao nao retornou conteudo")

    pergunta_rag = (
        "O usuário enviou uma imagem. O modelo de visão analisou a imagem e "
        "identificou o seguinte:\n"
        f"{conteudo}\n"
        "Fale sobre o assunto retratado na imagem: explique os conceitos envolvidos "
        "com base SOMENTE nos trechos do corpus fornecidos (contexto) e cite as "
        "fontes. Se o contexto não cobrir o assunto, responda exatamente NAO_SEI."
    )

    resultado = _pipeline_resposta(pergunta_rag)
    resultado["pergunta_rag"] = pergunta_rag
    resultado["transcricao"] = conteudo  # visao (nao exibida ao usuario; resposta e a do RAG)
    resultado["latencia_s"]["visao"] = round(time.time() - t_visao, 2)
    resultado["modelo_visao"] = meta_visao.get("modelo")
    return resultado


def _pipeline_resposta(texto: str, lat_asr: float | None = None) -> dict:
    """Texto -> (RAG/LLM) -> TTS. Compartilhado pelos fluxos de voz e imagem."""
    resposta = ""
    resultado_rag = None
    try:
        if os.getenv("USE_RAG", "true").lower() == "true":
            logger.debug("Usando RAG para resposta")
            # v0.3: usa o corpus texto + imagem quando processado; senao cai p/ texto (v0.2)
            if config.RAG_V3_CHUNK_PATH.exists():
                resultado_rag = rag_pipeline.responder_v3(texto)
            else:
                resultado_rag = rag_pipeline.responder(texto)
            resposta = resultado_rag["resposta"]
            meta_llm = {
                "modelo": resultado_rag["modelo_llm"],
                "uso": resultado_rag["uso"],
                "latencia_s": resultado_rag["latencia_s"],
            }
        else:
            resposta, meta_llm = llm.responder(texto)
    except Exception as e:
        logger.error("LLM/RAG falhou: {}. Usando resposta local.", e)
        resposta = "Desculpe, nao consegui consultar o modelo agora. Tente novamente."
        meta_llm = {"modelo": config.DEEPSEEK_MODEL, "uso": None, "latencia_s": 0.0}

    # TTS le apenas o conteudo (sem rodape de referencias nem marcadores [N]).
    tts_texto = resposta
    if isinstance(resultado_rag, dict):
        tts_texto = resultado_rag.get("resposta_tts") or resposta
    audio, lat_tts = tts.sintetizar(tts_texto)

    return {
        "transcricao": texto,
        "resposta": resposta,
        "audio": audio,
        "latencia_s": {
            "asr": round(lat_asr or 0.0, 2),
            "llm": meta_llm["latencia_s"],
            "tts": round(lat_tts, 2),
        },
    }


app = create_app()

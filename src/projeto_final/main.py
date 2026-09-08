"""FastAPI — API unificada de Voz, Imagem e Texto (v0.4).

Endpoint unico POST /chat: recebe audio, imagem OU texto; normaliza a entrada
para texto e roda o MESMO processo de RAG/resposta para todos os fluxos.
Saida: JSON {texto, resposta, audio_base64, ...}; audio somente quando a
entrada nao e texto puro.
"""

from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from starlette.concurrency import run_in_threadpool

from projeto_final import config, llm, tts, voz
from projeto_final.rag import pipeline as rag_pipeline

RAIZ = config.RAIZ
STATIC_INDEX = RAIZ / "static" / "index.html"
# v0.4 - o endpoint unico /chat aceita arquivo de audio OU de imagem
# (o tipo e detectado pela extensao) ou texto digitado no campo "texto".
CHAT_EXTS = (*voz.AUDIO_EXTS,)
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
        title="Assistente Master IAG & LLM - API unificada de Voz, Imagem e Texto",
        description="POST /chat: recebe audio, imagem ou texto; todas as entradas passam pelo mesmo RAG/multiagentes e a resposta e JSON {texto, resposta, audio_base64}. WAV em base64 apenas para audio/imagem (texto puro retorna audio_base64=null).",
        version="0.4.0",
    )

    @app.on_event("startup")
    def startup():
        logger.info("Iniciando API v0.4 (unificada) em {}", RAIZ)
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

    # v0.4 - API unificada: POST /chat recebe audio, imagem OU texto. A entrada
    # e normalizada para texto (_normalizar_entrada) e TODOS os fluxos passam
    # pelo mesmo processo de RAG/multiagentes + resposta (_pipeline_resposta).
    # WAV sintetizado (audio_base64) somente quando a entrada NAO e texto puro.

    @app.post("/chat")
    async def chat(file: UploadFile | None = File(None),
                   texto: str | None = Form(None)) -> JSONResponse:
        req = await _preparar_entrada(file, texto)
        t_total = time.time()
        try:
            entrada = await run_in_threadpool(_normalizar_entrada, req)
            resposta = await run_in_threadpool(
                _pipeline_resposta, entrada["pergunta_rag"],
                com_audio=(req["tipo"] != "texto"),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Erro no pipeline de {}: {}", req["tipo"], e)
            raise HTTPException(
                status_code=502,
                detail=f"Falha ao processar a entrada ({req['tipo']}): {e}",
            )
        logger.info(
            "Chat {} finalizado em {} s (modelo_entrada={}, com_audio={})",
            req["tipo"], round(time.time() - t_total, 2),
            entrada.get("modelo_entrada"), resposta.get("audio") is not None,
        )
        return JSONResponse(_montar_payload(req["tipo"], entrada, resposta))

    return app


async def _preparar_entrada(file: UploadFile | None,
                            texto: str | None) -> dict:
    """Valida a requisicao do /chat e devolve a entrada bruta.

    Regras: exatamente UMA das entradas — arquivo de audio/imagem (UploadFile)
    OU texto digitado. Arquivo -> tipo detectado pela extensao. Retorna dict:
    {"tipo": "audio"|"imagem"|"texto", "conteudo": bytes|str,
    "ext": str|None}.
    """
    tem_arquivo = file is not None
    tem_texto = bool((texto or "").strip())
    if tem_arquivo and tem_texto:
        raise HTTPException(
            status_code=400,
            detail="informe apenas UMA entrada: file (audio/imagem) ou texto",
        )
    if not tem_arquivo and not tem_texto:
        raise HTTPException(
            status_code=400,
            detail="informe file (audio/imagem) ou texto",
        )
    if tem_texto:
        return {"tipo": "texto", "conteudo": (texto or "").strip(), "ext": None}

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext in CHAT_EXTS:
        tipo = "audio"
    elif ext in IMAGEM_EXTS:
        tipo = "imagem"
    else:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Extensao nao suportada: .{ext} "
                f"(audio: {', '.join(CHAT_EXTS)}; "
                f"imagem: {', '.join(sorted(IMAGEM_EXTS))})"
            ),
        )
    dados = await file.read()
    if not dados:
        raise HTTPException(status_code=400, detail=f"{tipo} vazio")
    return {"tipo": tipo, "conteudo": dados, "ext": ext}


def _normalizar_entrada(req: dict) -> dict:
    """Converte a entrada bruta (audio/imagem/texto) em texto para o RAG.

    Etapa UNICA de normalizacao de entrada: audio -> transcricao ASR (local);
    imagem -> descricao do modelo de visao + prompt RAG ("fale sobre o assunto
    retratado"); texto -> as-is. Retorna dict com "texto" (exibido ao
    usuario), "pergunta_rag" (enviada ao RAG) e metadados de entrada
    (modelo_entrada, latencia_entrada_s).
    """
    tipo = req["tipo"]
    if tipo == "audio":
        prompt = config.ler_prompt_vocabulario()
        dados, ext = req["conteudo"], req["ext"]
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(dados)
            path = tmp.name
        try:
            texto, lat_asr = voz.transcrever(
                path, prompt=prompt, modelo=config.ASR_MODELO
            )
        finally:
            os.unlink(path)
        if not texto:
            logger.warning("Transcricao vazia")
            raise HTTPException(
                status_code=422, detail="Nao foi possivel transcrever o audio"
            )
        return {
            "texto": texto,
            "pergunta_rag": texto,
            "modelo_entrada": config.ASR_MODELO,
            "latencia_entrada_s": lat_asr,
        }

    if tipo == "imagem":
        mime = MIME_POR_EXT.get(req["ext"], "image/png")
        conteudo, meta_visao = llm.descrever_imagem(req["conteudo"], mime)
        if not conteudo:
            logger.warning("Modelo de visao nao retornou conteudo")
            raise HTTPException(
                status_code=502, detail="O modelo de visao nao retornou conteudo"
            )
        pergunta_rag = (
            "O usuário enviou uma imagem. O modelo de visão analisou a imagem e "
            "identificou o seguinte:\n"
            f"{conteudo}\n"
            "Fale sobre o assunto retratado na imagem: explique os conceitos envolvidos "
            "com base SOMENTE nos trechos do corpus fornecidos (contexto) e cite as "
            "fontes. Se o contexto não cobrir o assunto, responda exatamente NAO_SEI."
        )
        return {
            "texto": conteudo,
            "pergunta_rag": pergunta_rag,
            "modelo_entrada": meta_visao.get("modelo"),
            "latencia_entrada_s": meta_visao.get("latencia_s"),
        }

    # texto puro: ja e a pergunta final
    return {
        "texto": req["conteudo"],
        "pergunta_rag": req["conteudo"],
        "modelo_entrada": None,
        "latencia_entrada_s": None,
    }


def _montar_payload(tipo: str, entrada: dict, resposta: dict) -> dict:
    """Monta o JSON unificado da API a partir da entrada normalizada + resposta.

    Contrato v0.4: texto + audio (WAV em base64) sempre, EXCETO quando a entrada
    e apenas texto (audio_base64=null, latencia.tts=null).
    """
    lat = resposta.get("latencia_s") or {}
    audio = resposta.get("audio")
    return {
        "tipo_entrada": tipo,
        "texto": entrada["texto"],
        "resposta": resposta.get("resposta", ""),
        "audio_base64": (
            base64.b64encode(audio).decode("ascii") if audio else None
        ),
        "abstencao": resposta.get("abstencao"),
        "referencias": resposta.get("referencias") or [],
        "modelo_entrada": entrada.get("modelo_entrada"),
        "latencia": {
            "entrada": (
                round(entrada["latencia_entrada_s"], 2)
                if entrada.get("latencia_entrada_s") is not None else None
            ),
            "llm": lat.get("llm"),
            "tts": lat.get("tts"),
        },
    }


def _pipeline_resposta(texto: str, usar_agentes: bool | None = None,
                       com_audio: bool = True) -> dict:
    """(RAG/agentes) -> TTS (opcional). Etapa de RESPOSTA unica e compartilhada
    por audio/imagem/texto (a entrada ja foi normalizada para texto).

    v0.4: com AGENTES_HABILITADO (default true), a resposta passa pelo
    orquestrador multiagente (roteador SIMPLES/COMPLEXA + geradores com fallback)
    sempre no corpus texto+imagem (v0.3). Desligue com AGENTES_HABILITADO=false
    ou passar usar_agentes=False (volta ao RAG v0.3 de modelo unico).
    com_audio=False (entrada = texto puro) pula o TTS e devolve audio=None
    (mais referencias/abstencao no dict).
    """
    if usar_agentes is None:
        usar_agentes = config.AGENTES_HABILITADO
    resposta = ""
    resultado_rag = None
    try:
        if os.getenv("USE_RAG", "true").lower() == "true":
            logger.debug("Gerando resposta (agentes={}, corpus v0.3={})",
                         usar_agentes, config.RAG_V3_CHUNK_PATH.exists())
            # v0.3/v0.4: usa o corpus texto + imagem quando processado; senao cai p/ texto (v0.2)
            if config.RAG_V3_CHUNK_PATH.exists():
                if usar_agentes:
                    # v0.4 - fluxo multiagente (roteador + geradores com fallback)
                    from projeto_final import agentes as agentes_mod

                    resultado_rag = agentes_mod.responder_agentes(texto)
                    logger.info("Multiagente (chat): rota={} agente={} fallback={}",
                                resultado_rag.get("rota"), resultado_rag.get("agente"),
                                resultado_rag.get("fallback"))
                else:
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

    # v0.4 - enriquece a resposta com abstenção e referências quando vierem do RAG
    abstencao = None
    referencias = []
    if isinstance(resultado_rag, dict):
        abstencao = resultado_rag.get("abstencao")
        referencias = resultado_rag.get("referencias") or []

    audio = None
    lat_tts = None
    if com_audio:
        # TTS le apenas o conteudo (sem rodape de referencias nem marcadores [N]).
        tts_texto = resposta
        if isinstance(resultado_rag, dict):
            tts_texto = resultado_rag.get("resposta_tts") or resposta
        audio, lat_tts = tts.sintetizar(tts_texto)

    return {
        "resposta": resposta,
        "audio": audio,          # bytes WAV ou None (com_audio=False)
        "abstencao": abstencao,
        "referencias": referencias,
        "latencia_s": {
            "llm": meta_llm["latencia_s"],
            "tts": round(lat_tts, 2) if lat_tts is not None else None,
        },
    }


app = create_app()

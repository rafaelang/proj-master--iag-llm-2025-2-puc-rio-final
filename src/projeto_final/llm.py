"""Cliente LLM (DeepSeek) — v0.1."""

from __future__ import annotations

import time

from loguru import logger
from openai import OpenAI

from projeto_final import config

DEFAULT_MODELO = "deepseek-chat"
MAX_PALAVRAS = 30
MAX_TOKENS = 100
MAX_TOKENS_RAG = 400

NAO_SEI = "NAO_SEI"

PROMPT_TTS = (
    "Você é um assistente do Master IAG e LLM da PUC-Rio. "
    "Responda em português, em texto puro, SEM formatação: sem negrito, sem itálico, "
    "sem títulos, sem listas, sem marcadores, sem símbolos e sem emojis — "
    "apenas frases prontas para serem lidas em voz alta. "
    f"Seja direto e objetivo. Sua resposta deve ter no máximo {MAX_PALAVRAS} palavras "
    f"e no máximo {MAX_TOKENS} tokens, então responda de forma breve e compatível com esse limite."
)

_cliente_cache: OpenAI | None = None


def _cliente() -> OpenAI:
    global _cliente_cache
    if _cliente_cache is None:
        chave = config.DEEPSEEK_API_KEY
        if not chave or chave.startswith("sk-") is False:
            logger.warning("DEEPSEEK_API_KEY nao configurada corretamente")
            raise RuntimeError("DEEPSEEK_API_KEY nao configurada")
        _cliente_cache = OpenAI(api_key=chave, base_url=config.DEEPSEEK_BASE_URL)
        logger.debug("Cliente DeepSeek inicializado: {}", config.DEEPSEEK_MODEL)
    return _cliente_cache


def responder(pergunta: str, sistema: str | None = None) -> tuple[str, dict]:
    """Envia a pergunta ao LLM e retorna (resposta, metadados)."""
    t0 = time.time()
    modelo = config.DEEPSEEK_MODEL
    msgs: list[dict] = [
        {"role": "system", "content": sistema or PROMPT_TTS},
        {"role": "user", "content": pergunta},
    ]
    logger.debug("Chamando LLM com pergunta: {}", pergunta)
    try:
        resp = _cliente().chat.completions.create(
            model=modelo, messages=msgs, max_tokens=MAX_TOKENS
        )
    except Exception as e:
        logger.error("Erro na chamada LLM: {}", e)
        raise
    texto = (resp.choices[0].message.content or "").strip()
    uso = None
    if resp.usage is not None:
        uso = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    latencia = time.time() - t0
    logger.debug("LLM resposta: {} ({} s)", texto, round(latencia, 2))
    return texto, {"modelo": modelo, "uso": uso, "latencia_s": round(latencia, 2)}


def responder_com_contexto(pergunta: str, contexto: str, sistema: str | None = None) -> tuple[str, dict]:
    """Envia pergunta + contexto ao LLM e retorna (resposta, metadados)."""
    t0 = time.time()
    modelo = config.DEEPSEEK_MODEL
    if sistema is None:
        sistema = config.ler_prompt("v0.2/rag_sistema.txt") or ""
    msgs = [
        {"role": "system", "content": sistema},
        {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"},
    ]
    # Log do payload ENRIQUECIDO que sera enviado ao LLM (prompt system + contexto/chunks + pergunta)
    logger.debug("LLM RAG payload enviado ao modelo {} (prompt system + contexto + pergunta):", modelo)
    for _msg in msgs:
        logger.debug("  --- role: {} ({} chars) ---\n{}", _msg["role"], len(_msg["content"]), _msg["content"])
    try:
        resp = _cliente().chat.completions.create(
            model=modelo, messages=msgs, max_tokens=MAX_TOKENS_RAG
        )
    except Exception as e:
        logger.error("Erro na chamada LLM RAG: {}", e)
        raise
    texto = (resp.choices[0].message.content or "").strip()
    uso = None
    if resp.usage is not None:
        uso = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    latencia = time.time() - t0
    logger.debug("LLM RAG resposta: {} ({} s)", texto, round(latencia, 2))
    return texto, {"modelo": modelo, "uso": uso, "latencia_s": round(latencia, 2)}


def detectar_abstencao(resposta: str) -> bool:
    """Retorna True se a resposta indica abstenção (NAO_SEI)."""
    return NAO_SEI in resposta


# ------------------------------------------------------------- v0.3 - visao multimodal

PROMPT_VISAO = (
    "Você é o módulo de visão do assistente do Master IAG e LLM da PUC-Rio. "
    "Descreva a imagem recebida em português, de forma objetiva e completa: "
    "1) o que a imagem mostra (figura, diagrama, tabela, slide, foto); "
    "2) todo texto legível, termos, rótulos e relações entre elementos "
    "(ex.: colunas e tipos em um modelo, entidades de um diagrama ER); "
    "3) contexto visual relevante (cores, setas, hierarquia, partes numeradas). "
    "Seja fiel à imagem: não invente conteúdo que não esteja visível. "
    "Sem formatação markdown, sem listas com marcadores; texto puro corrido."
)


def descrever_imagem(dados: bytes, mime: str) -> tuple[str, dict]:
    """Envia uma imagem ao modelo de visao (DeepSeek) e retorna a descricao.

    A descricao vira a "transcricao" da imagem no chat e a pergunta do RAG.
    """
    import base64

    if not dados:
        raise ValueError("imagem vazia")
    modelo = config.DEEPSEEK_VISION_MODEL
    if not modelo:
        raise RuntimeError("DEEPSEEK_VISION_MODEL nao configurado")
    mime = mime or "image/png"
    b64 = base64.b64encode(dados).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    t0 = time.time()
    msgs = [
        {"role": "system", "content": PROMPT_VISAO},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Descreva esta imagem:"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    logger.debug("Chamando modelo de visao {} ({} bytes em base64)", modelo, len(b64))
    try:
        resp = _cliente().chat.completions.create(
            model=modelo, messages=msgs, max_tokens=config.DEEPSEEK_VISION_MAX_TOKENS
        )
    except Exception as e:
        logger.error("Erro na chamada do modelo de visao: {}", e)
        raise
    texto = (resp.choices[0].message.content or "").strip()
    uso = None
    if resp.usage is not None:
        uso = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    latencia = time.time() - t0
    logger.debug("Visao resposta: {} chars em {} s", len(texto), round(latencia, 2))
    return texto, {"modelo": modelo, "uso": uso, "latencia_s": round(latencia, 2)}

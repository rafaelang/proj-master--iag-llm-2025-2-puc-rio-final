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
    logger.debug("Chamando LLM RAG com contexto de {} chars", len(contexto))
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

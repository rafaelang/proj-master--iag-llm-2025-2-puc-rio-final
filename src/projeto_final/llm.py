"""Cliente LLM (DeepSeek) — v0.1."""

from __future__ import annotations

import re
import time
import unicodedata

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
    "Responda em português com a acentuação correta (ex.: não, informação, é), "
    "em texto puro, SEM formatação: sem negrito, sem itálico, "
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
    """Retorna True se a resposta indica abstenção.

    Insensível a acento/caixa e a variantes do token (NAO_SEI, NÃO_SEI,
    "não sei", "não há informações", "não tenho informações"). O token pode
    aparecer sozinho ou misturado com texto explicativo — qualquer ocorrência
    marca a abstenção.
    """
    if not resposta:
        return False
    r = unicodedata.normalize("NFKD", resposta.lower())
    r = "".join(c for c in r if not unicodedata.combining(c))
    r = r.replace("_", " ").replace("-", " ")
    r = re.sub(r"\s+", " ", r)
    if "nao sei" in r:
        return True
    return bool(
        re.search(r"\bnao (ha|tenho|existem?) informa", r)
        or "nao ha informacoes" in r
        or "nao tenho informacoes" in r
    )


# ------------------------------------------------------------- v0.3 - visao multimodal

PROMPT_VISAO = (
    "Você é o módulo de visão do assistente do Master IAG e LLM da PUC-Rio. "
    "Analise a imagem e retorne APENAS o que for útil para um sistema de busca "
    "em materiais didáticos (RAG), no formato abaixo (texto puro, sem markdown):\n"
    "ASSUNTO: <frase curta com o tema central da imagem>\n"
    "TERMOS: <termos-chave separados por vírgula — conceitos, acrônimos, nomes de "
    "entidades, rótulos e texto legível na imagem>\n"
    "SINTESE: <1 a 2 frases objetivas sobre o conteúdo principal: tipo de figura, "
    "relações entre elementos, o que a imagem ensina>\n"
    "Regras: IGNORE ruído visual (marca d'água, logos, URLs de banco de imagens, "
    "menus/barras de navegador, texto de interface). Seja fiel ao conteúdo visível: "
    "não invente termos. Responda DIRETAMENTE com o formato acima, sem preâmbulos "
    "e sem raciocínio longo."
)


def descrever_imagem(dados: bytes, mime: str) -> tuple[str, dict]:
    """Envia uma imagem ao modelo de visao (DeepSeek) e retorna a descricao.

    O modelo de visao e do tipo "reasoning": se a 1a chamada esgotar o orcamento
    de tokens so pensando (finish_reason='length' com content vazio), tenta uma
    segunda vez com orcamento maior antes de desistir.
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

    def _chamar(budget: int):
        logger.debug("Chamando modelo de visao {} (orcamento {} tokens)", modelo, budget)
        return _cliente().chat.completions.create(
            model=modelo, messages=msgs, max_tokens=budget
        )

    t0 = time.time()
    try:
        resp = _chamar(config.DEEPSEEK_VISION_MAX_TOKENS)
        texto = (resp.choices[0].message.content or "").strip()
        fin = resp.choices[0].finish_reason
        retry = False

        # Retry: raciocinou demais e nao respondeu (comum em imagens complexas).
        if not texto and fin == "length":
            logger.warning(
                "Visao esgotou {} tokens sem responder (finish_reason=length). "
                "Tentando com orcamento maior ({} tokens)...",
                config.DEEPSEEK_VISION_MAX_TOKENS, config.DEEPSEEK_VISION_MAX_TOKENS_RETRY,
            )
            resp = _chamar(config.DEEPSEEK_VISION_MAX_TOKENS_RETRY)
            texto = (resp.choices[0].message.content or "").strip()
            fin = resp.choices[0].finish_reason
            retry = True

        if not texto:
            raise RuntimeError(
                "o modelo de visao nao retornou descricao (finish_reason="
                f"{fin!r}). Aumente DEEPSEEK_VISION_MAX_TOKENS e/ou "
                "DEEPSEEK_VISION_MAX_TOKENS_RETRY no .env"
            )
    except Exception:
        raise

    uso = None
    if resp.usage is not None:
        detalhes = getattr(resp.usage, "completion_tokens_details", None)
        uso = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "reasoning_tokens": getattr(detalhes, "reasoning_tokens", None) if detalhes else None,
        }
    latencia = time.time() - t0
    logger.debug("Visao resposta: {} chars em {} s (retry={})", len(texto), round(latencia, 2), retry)
    return texto, {"modelo": modelo, "uso": uso, "latencia_s": round(latencia, 2), "retry": retry}

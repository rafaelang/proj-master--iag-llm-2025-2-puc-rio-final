"""SLM local — Qwen2.5-1.5B-Instruct (GGUF Q4_K_M) via llama-cpp-python, 100% CPU.

Papeis na v0.4 (Agentes): roteador SIMPLES/COMPLEXA e gerador da rota simples.
Custo: US$ 0.00 (100% local/offline). Carregamento LAZY: o modelo (~1,1 GB) so
entra em memoria na primeira chamada — evita pressionar o processo quando a rota
nao usa o SLM. O GGUF vive em data/processed/slm_models/ (fora do Git).
"""

from __future__ import annotations

import atexit
import time
from pathlib import Path

from loguru import logger

from projeto_final import config

_llm = None


def _fechar() -> None:
    """Fecha o SLM no encerramento do processo.

    Fechar enquanto os modulos ainda estao vivos evita o erro benigno do
    deallocator do llama-cpp-python no Python 3.14 (Llama.__del__ no shutdown).
    """
    global _llm
    if _llm is not None:
        try:
            _llm.close()
        except Exception:
            pass
        _llm = None


atexit.register(_fechar)


def _get_llm():
    """Instancia (uma unica vez) o Llama via llama-cpp-python."""
    global _llm
    if _llm is None:
        caminho = Path(config.SLM_MODELO_PATH)
        if not caminho.exists():
            raise FileNotFoundError(
                f"GGUF do SLM nao encontrado em {caminho}. Baixe o arquivo "
                "qwen2.5-1.5b-instruct-q4_k_m.gguf (Qwen/Qwen2.5-1.5B-Instruct-GGUF) "
                "para data/processed/slm_models/ (fora do Git)."
            )
        from llama_cpp import Llama

        logger.info("Carregando SLM local {} (pode levar alguns segundos)...", caminho.name)
        t0 = time.time()
        _llm = Llama(
            model_path=str(caminho),
            n_ctx=config.SLM_N_CTX,
            n_threads=config.SLM_N_THREADS,
            n_gpu_layers=config.SLM_N_GPU_LAYERS,
            verbose=False,
        )
        logger.info("SLM carregado em {:.1f}s", time.time() - t0)
    return _llm


def disponivel() -> bool:
    """True se o GGUF ja foi baixado para disco."""
    return Path(config.SLM_MODELO_PATH).exists()


def chat(messages: list[dict], max_tokens: int | None = None,
         temperature: float | None = None) -> str:
    """Gera texto com o SLM local.

    messages: [{"role": "system"|"user", "content": ...}]. Lanca excecao se o
    modelo nao estiver disponivel ou a geracao falhar (o orquestrador decide o
    fallback).
    """
    t0 = time.time()
    max_tokens = max_tokens or config.SLM_MAX_TOKENS
    temperature = config.SLM_TEMPERATURE if temperature is None else temperature
    resposta = _get_llm().create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # llama-cpp-python pode devolver dict (baixo nivel) ou objeto nativo
    if isinstance(resposta, dict):
        texto = (resposta.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    else:
        texto = getattr(resposta.choices[0].message, "content", None) or ""
    texto = texto.strip()
    logger.debug("SLM resposta: {} chars em {:.2f}s", len(texto), time.time() - t0)
    return texto

"""Configuracao central do projeto_final."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent.parent

# Carrega variaveis de ambiente do .env (se existir)
load_dotenv(RAIZ / ".env")
# Se o ambiente herdou a variavel vazia (ex.: shell com DEEPSEEK_API_KEY=), o
# python-dotenv nao sobrescreve por padrao — recarrega com override para garantir
# que o .env local (fonte canonica em dev) tenha precedencia quando o valor e vazio.
if not os.getenv("DEEPSEEK_API_KEY"):
    load_dotenv(RAIZ / ".env", override=True)

# Diretorios
DATA_DIR = RAIZ / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_SET_DIR = DATA_DIR / "golden_set"
PROMPTS_DIR = RAIZ / "prompts"
STATIC_DIR = RAIZ / "static"
DOCS_DIR = RAIZ / "docs"

# ASR
ASR_MODELO = os.getenv("ASR_MODELO", "small")
WHISPER_CACHE_DIR = PROCESSED_DIR / "whisper_models"
PROMPT_VOCABULARIO = PROMPTS_DIR / "v0.1" / "vocabulario_voz.txt"

# TTS
TTS_BACKEND = os.getenv("TTS_BACKEND", "piper")
PIPER_VOICE = os.getenv("PIPER_VOICE", "pt_BR-cadu-medium")
PIPER_MODEL = os.getenv("PIPER_MODEL")
PIPER_MODELS_DIR = PROCESSED_DIR / "piper_models"
# Prosodia do Piper (SynthesisConfig). length_scale > 1.0 desacelera a fala,
# reduzindo consoantes "atropeladas"/sílabas engolidas (defaults do Piper:
# length=1.0, noise=0.667, noise_w=0.8, volume=1.0).
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.1"))
PIPER_NOISE_SCALE = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
PIPER_NOISE_W_SCALE = float(os.getenv("PIPER_NOISE_W_SCALE", "0.8"))
PIPER_VOLUME = float(os.getenv("PIPER_VOLUME", "1.0"))

# RAG
RAG_DIR = PROCESSED_DIR / "rag"
RAG_CHUNK_PATH = RAG_DIR / "chunks.json"
RAG_BM25_PATH = RAG_DIR / "bm25.pkl"
RAG_EMBEDDINGS_PATH = RAG_DIR / "embeddings.npy"
RAG_CHUNK_IDS_PATH = RAG_DIR / "chunk_ids.json"
RAG_GOLDEN_SET = GOLDEN_SET_DIR / "rag" / "perguntas.json"
# v0.3 - Imagem/OCR: indice combinado (texto + imagem) em pasta propria (nao toca a v0.2)
RAG_V3_DIR = PROCESSED_DIR / "rag_v3"
RAG_V3_CHUNK_PATH = RAG_V3_DIR / "chunks.json"
RAG_V3_BM25_PATH = RAG_V3_DIR / "bm25.pkl"
RAG_V3_EMBEDDINGS_PATH = RAG_V3_DIR / "embeddings.npy"
RAG_V3_CHUNK_IDS_PATH = RAG_V3_DIR / "chunk_ids.json"
# v0.3 - imagens do corpus (extracao/OCR)
IMG_REGISTRO_PATH = RAG_V3_DIR / "imagens_registro.json"
IMG_OCR_PATH = RAG_V3_DIR / "imagens_ocr.json"
IMG_CHUNKS_PATH = RAG_V3_DIR / "imagens_chunks.json"
IMG_PAGINAS_PATH = RAG_V3_DIR / "paginas.json"
IMAGENS_DIR = PROCESSED_DIR / "imagens"
RAG_IMG_GOLDEN_SET = GOLDEN_SET_DIR / "imagem" / "perguntas.json"
IMG_MIN_LADO = int(os.getenv("IMG_MIN_LADO", "40"))  # filtra icones/logos minusculos
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
RERANK_CACHE_DIR = RAG_DIR / "rerank_models"
# Rerank cross-encoder: DESATIVADO por padrao (decisao medida na v0.2: MRR 0.906
# -> 0.865 e ~40 s/query em CPU). Para testar/ativar: RAG_RERANK=true.
RAG_RERANK = os.getenv("RAG_RERANK", "false").lower() == "true"

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
JUIZ_MODEL = os.getenv("JUIZ_MODEL", "deepseek-v4-pro")
# v0.3 - visao multimodal (descricao de imagens enviadas no chat)
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-v4-flash-vision-exp")
# O modelo de visao e "reasoning": o max_tokens limita raciocinio + resposta juntos.
# Imagens complexas podem gastar muito no raciocinio; default generoso evita
# respostas vazias (finish_reason='length' com content='').
DEEPSEEK_VISION_MAX_TOKENS = int(os.getenv("DEEPSEEK_VISION_MAX_TOKENS", "2000"))
# Tentativa de retry com orcamento maior quando a 1a chamada esgota sem responder.
DEEPSEEK_VISION_MAX_TOKENS_RETRY = int(os.getenv("DEEPSEEK_VISION_MAX_TOKENS_RETRY", "6000"))

# Rate limiting (por IP do cliente, janela deslizante em memoria)
RATELIMIT_HABILITADO = os.getenv("RATELIMIT_HABILITADO", "true").lower() == "true"
RATELIMIT_RAG_QTD = int(os.getenv("RATELIMIT_RAG_QTD", "4"))       # /rag/* e /chat* (geram LLM/custo)
RATELIMIT_GERAL_QTD = int(os.getenv("RATELIMIT_GERAL_QTD", "300"))  # "/" (pagina) e demais
RATELIMIT_PERIODO_S = int(os.getenv("RATELIMIT_PERIODO_S", "60"))

# v0.4 - SLM local (rota simples dos agentes): Qwen2.5-1.5B-Instruct GGUF Q4_K_M (~1,1 GB)
SLM_DIR = PROCESSED_DIR / "slm_models"
SLM_MODELO_PATH = os.getenv(
    "SLM_MODELO_PATH", str(SLM_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf")
)
SLM_N_THREADS = int(os.getenv("SLM_N_THREADS", "8"))
SLM_N_CTX = int(os.getenv("SLM_N_CTX", "4096"))
SLM_MAX_TOKENS = int(os.getenv("SLM_MAX_TOKENS", "400"))
SLM_TEMPERATURE = float(os.getenv("SLM_TEMPERATURE", "0.2"))

# v0.4 - Agentes (roteador SIMPLES/COMPLEXA + geradores com fallback).
# Mapeamento: slm = local · flash = DEEPSEEK_MODEL (deepseek-chat, usado na v0.3) ·
# pro = JUIZ_MODEL (deepseek-v4-pro, usado na v0.3). Modelos escolhiveis por env/CLI.
AGENTE_MODELO_FLASH = os.getenv("AGENTE_MODELO_FLASH", DEEPSEEK_MODEL)
AGENTE_MODELO_PRO = os.getenv("AGENTE_MODELO_PRO", JUIZ_MODEL)
AGENTE_ROTEADOR = os.getenv("AGENTE_ROTEADOR", "slm")
AGENTE_SIMPLES = os.getenv("AGENTE_SIMPLES", "slm")
AGENTE_COMPLEXA = os.getenv("AGENTE_COMPLEXA", "pro")
# deepseek-v4-pro e "reasoning": o max_tokens limita raciocinio + resposta juntos.
# Orcamento generoso evita resposta vazia (finish_reason='length') no gerador pro.
AGENTE_PRO_MAX_TOKENS = int(os.getenv("AGENTE_PRO_MAX_TOKENS", "1500"))

def ler_prompt(caminho_relativo: str) -> str | None:
    """Le um prompt versionado em prompts/ por caminho relativo."""
    p = PROMPTS_DIR / caminho_relativo
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def ler_prompt_vocabulario() -> str | None:
    """Le o vocabulario de dominio para initial_prompt do Whisper."""
    return ler_prompt("v0.1/vocabulario_voz.txt")

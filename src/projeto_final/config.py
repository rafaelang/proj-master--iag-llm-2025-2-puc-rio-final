"""Configuracao central do projeto_final."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent.parent

# Carrega variaveis de ambiente do .env (se existir)
load_dotenv(RAIZ / ".env")

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
PIPER_VOICE = os.getenv("PIPER_VOICE", "pt_BR-faber-medium")
PIPER_MODEL = os.getenv("PIPER_MODEL")
PIPER_MODELS_DIR = PROCESSED_DIR / "piper_models"

# RAG
RAG_DIR = PROCESSED_DIR / "rag"
RAG_CHUNK_PATH = RAG_DIR / "chunks.json"
RAG_BM25_PATH = RAG_DIR / "bm25.pkl"
RAG_EMBEDDINGS_PATH = RAG_DIR / "embeddings.npy"
RAG_CHUNK_IDS_PATH = RAG_DIR / "chunk_ids.json"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

def ler_prompt(caminho_relativo: str) -> str | None:
    """Le um prompt versionado em prompts/ por caminho relativo."""
    p = PROMPTS_DIR / caminho_relativo
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def ler_prompt_vocabulario() -> str | None:
    """Le o vocabulario de dominio para initial_prompt do Whisper."""
    return ler_prompt("v0.1/vocabulario_voz.txt")

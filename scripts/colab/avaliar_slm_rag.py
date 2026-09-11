"""v0.5b · A/B — RAG com SLM base x SLM destilado (LoRA) no golden set (projeto_final).

Roda no Colab (T4). Reimplementado sobre os módulos do projeto_final (bm25 e
detecção de abstenção portados de forma self-contained — sem deps pesadas).

Uso (no Colab):
  python avaliar_slm_rag.py                      # SLM base + RAG
  python avaliar_slm_rag.py --no-rag             # SLM base, sem RAG (memória)
  python avaliar_slm_rag.py /content/data/processed/slm_adapter          # destilado + RAG
  python avaliar_slm_rag.py /content/data/processed/slm_adapter --no-rag # destilado, sem RAG

Variáveis de ambiente (v0.5b):
  PERGUNTAS_PATH=/content/data/golden_set/rag/perguntas_v05b.json  # golden expandido
  NOMEROLE=base|destilado_no_rag         # sufixo do arquivo de saída (default derivado)

Arquivos esperados (upload): /content/data/golden_set/rag/perguntas.json,
/content/data/processed/rag/chunks.json, /content/prompts/rag_sistema.txt.
Saída: /content/resultado_{base|destilado}[_no_rag].json
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import unicodedata

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER = None
NO_RAG = False
for arg in sys.argv[1:]:
    if arg == "--no-rag":
        NO_RAG = True
    elif not arg.startswith("--"):
        ADAPTER = arg

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNKS_PATH = "/content/data/processed/rag/chunks.json"
PERGUNTAS_PATH = os.environ.get(
    "PERGUNTAS_PATH", "/content/data/golden_set/rag/perguntas.json")
NOMEROLE = os.environ.get("NOMEROLE", "")
RAG_SYS = "/content/prompts/v0.2/rag_sistema.txt"
PROMPT_ONLY_SYS = "Você é um assistente. Responda à pergunta. Se não souber a resposta, responda apenas: Não sei."


# ------------------------------------------------------- bm25 (porta do projeto_final.bm25)

STOPWORDS_PT = frozenset({
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "do", "dos",
    "e", "em", "entre", "era", "essa", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "foram", "ha", "isso", "isto", "ja", "la", "mais", "mas",
    "meu", "minha", "muito", "na", "nao", "nas", "nem", "no", "nos", "nossa", "nosso",
    "o", "os", "ou", "para", "pela", "pelo", "pelas", "pelos", "por", "qual", "quando",
    "que", "quem", "se", "sem", "sendo", "ser", "seu", "sua", "so", "sobre", "te",
    "tem", "ter", "um", "uma", "umas", "uns", "voce",
})


def normalizar(texto: str) -> list[str]:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [tok for tok in t.split() if len(tok) > 1]


def preparar_query(pergunta: str) -> list[str]:
    return [tok for tok in normalizar(pergunta) if tok not in STOPWORDS_PT]


class BM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        df: dict[str, int] = {}
        for doc in corpus:
            for termo in set(doc):
                df[termo] = df.get(termo, 0) + 1
        n = len(corpus)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for termo in query:
            idf = self.idf.get(termo)
            if idf is None:
                continue
            for i, doc in enumerate(self.corpus):
                tf = doc.count(termo)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1.0) / denom
        return scores


# ------------------------------------------------------- abstencao (porta do projeto_final.llm)

def detectar_abstencao(resposta: str) -> bool:
    if not resposta:
        return False
    r = unicodedata.normalize("NFKD", resposta.lower())
    r = "".join(c for c in r if not unicodedata.combining(c))
    r = r.replace("_", " ").replace("-", " ")
    r = re.sub(r"\s+", " ", r)
    if "nao sei" in r:
        return True
    return bool(re.search(r"\bnao (ha|tenho|existem?) informa", r)
                or "nao ha informacoes" in r or "nao tenho informacoes" in r)


# ------------------------------------------------------- modelo

def carregar(adapter):
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


def gerar(model, tok, system, pergunta, contexto):
    user = f"PERGUNTA: {pergunta}\n\nEVIDÊNCIAS:\n{contexto}" if contexto else pergunta
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(txt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _montar_contexto(chunks, selecionados):
    linhas = []
    for i, c in enumerate(selecionados, start=1):
        linhas.append(f"[{i}] {c['doc_id']}, pagina {c['pagina']}: {c['texto']}")
    return "\n\n".join(linhas)


def main():
    nome = ("DESTILADO" if ADAPTER else "BASE") + ("-NO-RAG" if NO_RAG else "")
    chunks = json.load(open(CHUNKS_PATH)) if not NO_RAG else None
    bm25 = None
    if chunks:
        bm25 = BM25Okapi([normalizar(c["texto"]) for c in chunks])
    system = open(RAG_SYS).read() if not NO_RAG else PROMPT_ONLY_SYS
    perguntas = json.load(open(PERGUNTAS_PATH))["perguntas"]
    model, tok = carregar(ADAPTER)
    print(f"[{nome}] modelo carregado.")

    resultados = []
    for i, q in enumerate(perguntas):
        qid = q["id"]
        print(f"  [{nome}] #{qid:02d} {'recuperando (BM25)...' if chunks else 'sem evidências...'}", flush=True)
        ctx = ""
        if chunks:
            scores = bm25.get_scores(preparar_query(q["pergunta"]))
            top = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:5]
            selecionados = [chunks[j] for j in top if scores[j] > 0][:5]
            ctx = _montar_contexto(chunks, selecionados)
        print(f"  [{nome}] #{qid:02d} gerando...", flush=True)
        resp = gerar(model, tok, system, q["pergunta"], ctx)
        absteve = detectar_abstencao(resp)
        deve = q.get("deve_abster", False)
        resultados.append({"id": qid, "pergunta": q["pergunta"], "absteve": absteve,
                           "deve": deve, "ok": absteve == deve, "resposta": resp[:300]})
        print(f"  [{nome}] #{qid:02d} OK ({len(resp)} chars, absteve={absteve})", flush=True)

    n = len(resultados)
    ok = sum(r["ok"] for r in resultados)
    print(f"[{nome}] abstenção correta: {ok}/{n} = {ok/n:.2f}")
    for r in resultados:
        flag = "OK " if r["ok"] else "FALHA"
        print(f"  [{flag}] #{r['id']:02d} absteve={r['absteve']} deve={r['deve']} | {r['pergunta'][:45]}")

    nome_arq = NOMEROLE or (("destilado" if ADAPTER else "base") + ("_no_rag" if NO_RAG else ""))
    json.dump(resultados, open(f"/content/resultado_{nome_arq}.json", "w"), ensure_ascii=False, indent=2)
    print(f"salvo: resultado_{nome_arq}.json")


if __name__ == "__main__":
    main()

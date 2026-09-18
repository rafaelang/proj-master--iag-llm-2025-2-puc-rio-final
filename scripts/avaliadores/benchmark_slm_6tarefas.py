"""v1.0 · Fase 2 — Bateria de Eval SLM (6 tarefas da aula08).

Mede o SLM local (Qwen 0.5B × 1.5B × 3B GGUF) nas seis tarefas do deck da Aula 08
(`duelo_modelos_seis_tarefas`), com a régua de produção do projeto:

  1. Estreita      — pergunta factual com contexto (triagem) → resposta direta
  2. JSON output   — contrato de saída estruturada (parsing determinístico)
  3. Recusa honesta— pergunta fora do corpus → deve abster (NAO_SEI)
  4. Raciocínio    — pergunta composta que exige síntese
  5. Juiz          — escolha de preferência A/B (concordância com sinal de ouro)
  6. Comprimento   — resposta sob teto de tokens

Cada tarefa usa um subconjunto do golden v0.6 (132Q) filtrado por estrato, para que
as 6 dimensões sejam avaliadas no mesmo material de produção. Compara os dois modelos
intra-sessão (mesma máquina, mesmo seed, mesmo prompt).

Uso (Colab/local):
  python scripts/avaliadores/benchmark_slm_6tarefas.py --modelo 1.5b    # default
  python scripts/avaliadores/benchmark_slm_6tarefas.py --modelo 0.5b
  python scripts/avaliadores/benchmark_slm_6tarefas.py --modelo 3b
  python scripts/avaliadores/benchmark_slm_6tarefas.py --modelo 1.5b --max-q 20

Saída: data/processed/v10_bateria/duelo_<modelo>.jsonl + _resumo.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from loguru import logger

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from dotenv import load_dotenv

load_dotenv(RAIZ / ".env", override=True)

from projeto_final import config

GGUF_30 = config.SLM_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
GGUF_15 = config.SLM_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
GGUF_05 = config.SLM_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODELOS = {"3b": GGUF_30, "1.5b": GGUF_15, "0.5b": GGUF_05}

GOLDEN = config.GOLDEN_SET_DIR / "rag" / "perguntas_v06.json"
SAIDA_DIR = config.PROCESSED_DIR / "v10_bateria"

SEED = 42
MAX_Q = 132

# Prompt de sistema de produção (SLM rota simples) — mesma régua do cascade.
SISTEMA = (config.PROMPTS_DIR / "v0.4" / "rag_sistema_slm.txt").read_text(encoding="utf-8")


def _llm(modelo: str, n_gpu_layers: int = 99):
    from llama_cpp import Llama

    return Llama(model_path=str(MODELOS[modelo]), n_gpu_layers=n_gpu_layers,
                 n_ctx=2048, verbose=False, seed=SEED)


def _perguntas() -> list[dict]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]


# ---------------------------------------------------------------------------
# Tarefas — cada uma recebe (pergunta, contexto) e devolve (pred, meta)
# ---------------------------------------------------------------------------

def tarefa_estreita(p: dict, ctx: str, model) -> dict:
    """Triagem factual: resposta curta e direta com contexto."""
    msgs = [{"role": "system", "content": SISTEMA},
            {"role": "user", "content": f"CONTEXTO:\n{ctx}\n\nPERGUNTA: {p['pergunta']}\nResposta:"}]
    t0 = time.time()
    r = model.create_chat_completion(messages=msgs, max_tokens=80, temperature=0.0)
    return {"pred": r["choices"][0]["message"]["content"].strip(),
            "latencia_s": time.time() - t0,
            "uso": r.get("usage", {})}


def tarefa_json(p: dict, ctx: str, model) -> dict:
    """Contrato de saída JSON: extrai o dado estruturado do contexto."""
    prompt = (f"CONTEXTO:\n{ctx}\n\nPERGUNTA: {p['pergunta']}\n"
              "Responda APENAS com um JSON válido {\"resposta\": <texto>, \"citacao\": <n>}.")
    msgs = [{"role": "system", "content": SISTEMA},
            {"role": "user", "content": prompt}]
    t0 = time.time()
    r = model.create_chat_completion(messages=msgs, max_tokens=120, temperature=0.0)
    txt = r["choices"][0]["message"]["content"].strip()
    # parse JSON (tolera markdown fenced)
    ok, dados = False, None
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if m:
        try:
            dados = json.loads(m.group())
            ok = isinstance(dados, dict) and "resposta" in dados
        except Exception:
            ok = False
    return {"pred": txt, "json_valido": ok, "dados": dados,
            "latencia_s": time.time() - t0, "uso": r.get("usage", {})}


def tarefa_recusa(p: dict, ctx: str, model) -> dict:
    """Recusa honesta: adversarial → deve abster (NAO_SEI)."""
    msgs = [{"role": "system", "content": SISTEMA},
            {"role": "user", "content": f"CONTEXTO:\n{ctx}\n\nPERGUNTA: {p['pergunta']}\nResposta:"}]
    t0 = time.time()
    r = model.create_chat_completion(messages=msgs, max_tokens=80, temperature=0.0)
    txt = r["choices"][0]["message"]["content"].strip()
    absteve = "NAO_SEI" in txt.upper() or "não sei" in txt.lower()
    return {"pred": txt, "absteve": absteve, "latencia_s": time.time() - t0,
            "uso": r.get("usage", {})}


def tarefa_raciocinio(p: dict, ctx: str, model) -> dict:
    """Pergunta composta: exige síntese de múltiplos trechos."""
    msgs = [{"role": "system", "content": SISTEMA},
            {"role": "user", "content": f"CONTEXTO:\n{ctx}\n\nPERGUNTA: {p['pergunta']}\nResposta:"}]
    t0 = time.time()
    r = model.create_chat_completion(messages=msgs, max_tokens=200, temperature=0.2)
    return {"pred": r["choices"][0]["message"]["content"].strip(),
            "latencia_s": time.time() - t0, "uso": r.get("usage", {})}


def tarefa_juiz(p: dict, ctx: str, model) -> dict:
    """Juiz de preferência: escolhe entre resposta COM citação e SEM citação.

    O par é determinístico: A = resposta com citação [1] (a "boa" na régua do
    projeto, que exige fonte) e B = resposta curta sem citação. Mede (a) se o
    juiz prefere a resposta fundamentada e (b) viés de posição invertendo A/B.
    """
    boa = "De acordo com o material [1], a resposta é baseada no contexto do curso."
    ruim = "A resposta é a que eu acho correta, sem precisar de fonte."
    # ordem normal: A=boa, B=ruim · ordem invertida: A=ruim, B=boa
    for ordem, rotulo in [("normal", "AB"), ("invertida", "BA")]:
        if ordem == "normal":
            a, b = boa, ruim
        else:
            a, b = ruim, boa
        prompt = (f"PERGUNTA: {p['pergunta']}\n"
                  f"Duas respostas:\nA: {a}\nB: {b}\n"
                  "Qual é a melhor para um assistente que deve citar a fonte? "
                  "Responda APENAS com A ou B.")
        msgs = [{"role": "system", "content": SISTEMA},
                {"role": "user", "content": prompt}]
        t0 = time.time()
        r = model.create_chat_completion(messages=msgs, max_tokens=8, temperature=0.0)
        txt = r["choices"][0]["message"]["content"].strip()
        esc = re.search(r"[AB]", txt)
        escolha = esc.group() if esc else None
        # converte para "boa"/"ruim" independente da ordem
        pref = "boa" if (escolha == "A") == (ordem == "normal") else "ruim"
        lat = time.time() - t0
        if ordem == "normal":
            normal = {"pred": txt, "escolha_bruta": escolha, "preferencia": pref,
                      "latencia_s": lat, "uso": r.get("usage", {})}
        else:
            invertida = {"pred": txt, "escolha_bruta": escolha, "preferencia": pref,
                         "latencia_s": lat, "uso": r.get("usage", {})}
    # viés: preferência mudou quando inverteu?
    vies = normal["preferencia"] != invertida["preferencia"]
    return {"pred": normal["pred"], "escolha_normal": normal["escolha_bruta"],
            "escolha_invertida": invertida["escolha_bruta"],
            "preferencia_normal": normal["preferencia"],
            "preferencia_invertida": invertida["preferencia"],
            "vies_posicao": vies,
            "latencia_s": (normal["latencia_s"] or 0) + (invertida["latencia_s"] or 0),
            "uso": None}


def tarefa_comprimento(p: dict, ctx: str, model) -> dict:
    """Comprimento: resposta sob teto de 30 tokens."""
    msgs = [{"role": "system", "content": SISTEMA},
            {"role": "user", "content": (f"CONTEXTO:\n{ctx}\n\nPERGUNTA: {p['pergunta']}\n"
                                         "Responda em ATÉ 30 palavras.")}]
    t0 = time.time()
    r = model.create_chat_completion(messages=msgs, max_tokens=50, temperature=0.2)
    txt = r["choices"][0]["message"]["content"].strip()
    return {"pred": txt, "n_tokens": len(txt.split()), "latencia_s": time.time() - t0,
            "uso": r.get("usage", {})}


TAREFAS = {
    "estreita": {"fn": tarefa_estreita, "estrato": "rotineira", "n": 12},
    "json": {"fn": tarefa_json, "estrato": "rotineira", "n": 12},
    "recusa": {"fn": tarefa_recusa, "estrato": "adversarial", "n": 12},
    "raciocinio": {"fn": tarefa_raciocinio, "estrato": "composta", "n": 12},
    "juiz": {"fn": tarefa_juiz, "estrato": "rotineira", "n": 12},
    "comprimento": {"fn": tarefa_comprimento, "estrato": "composta", "n": 12},
}


def _ctx(p: dict) -> str:
    """Contexto de produção: trecho do chunk do doc esperado (ou corpus raso)."""
    # Em produção o contexto vem do RAG. Aqui usamos um trecho representativo:
    # o próprio material é demasiado grande para inline; usamos os primeiros
    # chunks do corpus para simular um contexto RAG típico (top-5 raso).
    try:
        from projeto_final.rag.pipeline import carregar_chunks
        chunks = carregar_chunks()
    except Exception:
        return "Trecho do material do curso sobre o tema da pergunta."
    # Seleciona chunks cujo texto mencione alguma palavra-chave da pergunta
    tokens = re.findall(r"\w{4,}", p["pergunta"].lower())
    hits = [c for c in chunks if any(t in c["texto"].lower() for t in tokens[:6])][:5]
    if not hits:
        hits = chunks[:5]
    return "\n---\n".join(f"[{i+1}] {c['texto'][:600]}" for i, c in enumerate(hits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["3b", "1.5b", "0.5b"], default="1.5b")
    ap.add_argument("--max-q", type=int, default=MAX_Q)
    ap.add_argument("--gpu", type=int, default=99)
    a = ap.parse_args()

    perguntas = _perguntas()
    rng = random.Random(SEED)
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = SAIDA_DIR / f"duelo_{a.modelo}.jsonl"
    log = open(jsonl, "a")
    resumo = {}

    model = _llm(a.modelo, a.gpu)
    logger.info("Modelo {} carregado ({})", a.modelo, MODELOS[a.modelo].name)

    for nome, spec in TAREFAS.items():
        # amostra estrato com seed fixa
        pool = [p for p in perguntas if p["estrato"] == spec["estrato"] and p["id"] <= a.max_q]
        amostra = rng.sample(pool, min(spec["n"], len(pool)))
        linhas = []
        for p in amostra:
            ctx = _ctx(p)
            try:
                r = spec["fn"](p, ctx, model)
            except Exception as e:
                logger.error("tarefa {} #{} falhou: {}", nome, p["id"], e)
                r = {"pred": "", "latencia_s": None, "uso": None}
            linha = {"tarefa": nome, "id": p["id"], "estrato": p["estrato"],
                     "pergunta": p["pergunta"][:200], **r}
            linhas.append(linha)
            log.write(json.dumps(linha, ensure_ascii=False) + "\n")
            log.flush()
        # métricas da tarefa
        m = {}
        if nome == "json":
            m = {"json_valido": sum(1 for l in linhas if l.get("json_valido")) / len(linhas)}
        elif nome == "recusa":
            m = {"abstencao_correta": sum(1 for l in linhas if l.get("absteve")) / len(linhas)}
        elif nome == "comprimento":
            over = sum(1 for l in linhas if (l.get("n_tokens") or 99) > 30)
            m = {"dentro_30_tokens": 1 - over / len(linhas),
                 "tokens_medios": round(sum(l.get("n_tokens") or 0 for l in linhas) / len(linhas), 1)}
        elif nome == "juiz":
            n_vies = sum(1 for l in linhas if l.get("vies_posicao"))
            pref_boa = sum(1 for l in linhas if l.get("preferencia_normal") == "boa")
            m = {"prefere_boa_normal": pref_boa / len(linhas),
                 "vies_posicao": n_vies / len(linhas)}
        lat = [l["latencia_s"] for l in linhas if l.get("latencia_s")]
        m["latencia_media_s"] = round(sum(lat) / len(lat), 2) if lat else None
        resumo[nome] = {"n": len(linhas), **m}
        logger.info("{}: {}", nome, m)

    (SAIDA_DIR / f"duelo_{a.modelo}_resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    log.close()
    print("\n=== RESUMO", a.modelo, "===")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
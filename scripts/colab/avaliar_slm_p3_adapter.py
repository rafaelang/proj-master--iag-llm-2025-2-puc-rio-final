"""P3 · Lab — avaliar o SLM LORA (adapter) na rota simples (34Q), mesma schema do AB.

Igual ao avaliar_slm_p3.py, porem o gerador e o modelo HF em 4-bit + adaptador
LoRA (transformers/peft) em vez do GGUF via llama.cpp. Prompt de sistema FIXO
v0.4/rag_sistema_slm.txt (producao) para isolar o efeito do LoRA vs v04/p3a/p3b.

Roda no Colab (T4). Uso:
  RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/colab/avaliar_slm_p3_adapter.py ADAPTER_DIR

Saida: /content/data/processed/v05b_ab/slm_p3_lora.jsonl + _resumo.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata

import torch
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "/content/data/processed/slm_adapter_p3"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
GOLDEN = "/content/data/golden_set/rag/perguntas_v05b.json"
SAIDA = "/content/data/processed/v05b_ab/slm_p3_lora.jsonl"
PROMPT = "/content/prompts/v0.4/rag_sistema_slm.txt"

FROZEN_SIMPLES_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 20, 21, 22, 23, 24,
                      26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44]


def detectar_abstencao(resposta: str) -> bool:
    if not resposta:
        return False
    r = unicodedata.normalize("NFKD", resposta.lower())
    r = "".join(c for c in r if not unicodedata.combining(c))
    r = r.replace("_", " ").replace("-", " ")
    if "nao sei" in r:
        return True
    return bool(re.search(r"\bnao (ha|tenho|existem?) informa", r)
                or "nao ha informacoes" in r or "nao tenho informacoes" in r)


def main() -> None:
    sys.path.insert(0, "/content/src")
    from projeto_final import agentes, config
    from projeto_final.llm import detectar_abstencao as _det
    from projeto_final.rag.pipeline import carregar_chunks
    sys.path.insert(0, "/content/scripts/avaliadores")
    from avaliar_v5 import juiz

    _ = _det
    print(f"[lora] adapter={ADAPTER}")
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print("[lora] modelo + adapter carregados.")

    system = open(PROMPT).read().strip()
    perguntas = [q for q in json.load(open(GOLDEN))["perguntas"]
                 if int(q["id"]) in FROZEN_SIMPLES_IDS]
    perguntas.sort(key=lambda q: int(q["id"]))
    chunks = carregar_chunks()
    cliente = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    linhas = []
    for q in perguntas:
        t0 = time.time()
        qid = q["id"]
        recuperados = agentes.recuperar(q["pergunta"], chunks, top_k=5, base=None)
        contexto = agentes._formatar_contexto(recuperados) if recuperados else ""
        user = f"Contexto:\n{contexto}\n\nPergunta: {q['pergunta']}"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=400, do_sample=True, temperature=0.2,
                top_p=0.95, pad_token_id=tok.eos_token_id)
        resposta = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        absteve = detectar_abstencao(resposta)
        esperados = sorted(q.get("docs_esperados", []))
        try:
            j = juiz(q["pergunta"], esperados, resposta, cliente)
        except Exception as e:
            print(f"  [lora] #{qid} juiz erro: {e}")
            j = None
        linha = {
            "id": qid, "estrato": q["estrato"], "pergunta": q["pergunta"],
            "deve_abster": q.get("deve_abster", False), "docs_esperados": esperados,
            "absteve": absteve, "resposta": resposta[:400],
            "latencia_s": round(time.time() - t0, 2), "motivo": "lora",
            "prompt_src": "v0.4/rag_sistema_slm.txt", "juiz": j,
        }
        linhas.append(linha)
        print(f"#{qid:02d} absteve={absteve} deve={linha['deve_abster']} "
              f"juiz={j and j['correta']} aluc={(j or {}).get('alucinou')} "
              f"t={linha['latencia_s']}s", flush=True)

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    julg = [r for r in linhas if r.get("juiz")]
    acerto = sum(1 for r in julg if r["juiz"]["correta"])
    aluc = sum(1 for r in julg if r["juiz"]["alucinou"])
    abs_ok = sum(1 for r in linhas if r["absteve"] == r["deve_abster"])
    resumo = {
        "n": len(linhas), "acuracia": round(acerto / max(1, len(julg)), 3),
        "acertos": acerto, "julgadas": len(julg),
        "abstencao_correta": round(abs_ok / len(linhas), 3),
        "alucinacoes": aluc,
        "nota_media": round(sum(r["juiz"]["nota"] for r in julg) / max(1, len(julg)), 2),
        "n_jui_none": sum(1 for r in linhas if not r.get("juiz")),
    }
    print("\n=== P3 [lora] RESUMO ===")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    json.dump(resumo, open(SAIDA.replace(".jsonl", "_resumo.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[lora] salvo em {SAIDA}")


if __name__ == "__main__":
    main()
"""P3 · Lab — LoRA do SLM (Qwen2.5-1.5B-Instruct) no dataset REBALANCEADO da rota simples.

Diferenças vs train_slm.py (v0.5):
  - formato com CONTEXTO: "Contexto:\\n{ctx}\\n\\nPergunta: {q}\\nResposta: {a}",
    alinhado a inferencia do _gerar_slm (prompt rag_sistema_slm.txt);
  - perda apenas na RESPOSTA (labels -100 no trecho de contexto/pergunta);
  - max_length 2048 (top-5 do RAG cabe);
  - dataset: data/golden_set/adaptacao/dataset_sintetico_p3.json (rebalanceado).

Roda na T4 (Colab): 4-bit (bitsandbytes) + LoRA (peft), r=8, 4 epocas.
Saida: /content/data/processed/slm_adapter_p3
"""
from __future__ import annotations

import json
import os

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

MODELO = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET = os.environ.get("TRAIN_DATASET_P3",
                         "/content/data/golden_set/adaptacao/dataset_sintetico_p3.json")
SAIDA = os.environ.get("TRAIN_SAIDA_P3", "/content/data/processed/slm_adapter_p3")
MAX_LEN = int(os.environ.get("TRAIN_MAX_LEN_P3", "2048"))


def main() -> None:
    print(f"[train] carregando {MODELO} em 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(MODELO, quantization_config=bnb, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    tokenizer.pad_token = tokenizer.eos_token
    print("[train] modelo carregado.")

    dados = json.load(open(DATASET))
    n_ans = sum(1 for d in dados if d.get("tipo") == "answer")
    n_abs = sum(1 for d in dados if d.get("tipo") == "abstencao")
    print(f"[train] dataset p3: {len(dados)} itens (resposta={n_ans}, abstencao={n_abs})")

    def preparar(ex):
        ctx = (ex.get("contexto") or "").strip()
        q = (ex["pergunta"] or "").strip()
        a = (ex["resposta"] or "").strip()
        # reserva espaco para pergunta+resposta: limita o contexto por tokens (decodifica os ids
        # truncados — o texto final e ~MAX_LEN-256 tokens, entao prompt+resposta sempre cabem).
        max_ctx = MAX_LEN - 256
        if max_ctx > 0:
            ctx_ids = tokenizer(ctx, truncation=True, max_length=max_ctx)["input_ids"]
            ctx = tokenizer.decode(ctx_ids, skip_special_tokens=True)
        prompt = f"Contexto:\n{ctx}\n\nPergunta: {q}\nResposta:"
        full = f"{prompt} {a}"
        tok_p = tokenizer(prompt, truncation=True, max_length=MAX_LEN)
        tok_f = tokenizer(full, truncation=True, max_length=MAX_LEN)
        n_p = len(tok_p["input_ids"])
        input_ids = tok_f["input_ids"]
        labels = [-100] * min(n_p, len(input_ids)) + input_ids[min(n_p, len(input_ids)):]
        # pad determinístico ate MAX_LEN (o collator nao precisa re-padronizar)
        pad_tokens = MAX_LEN - len(input_ids)
        if pad_tokens < 0:
            input_ids, labels = input_ids[:MAX_LEN], labels[:MAX_LEN]
        else:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_tokens
            labels = labels + [-100] * pad_tokens
        return {"input_ids": input_ids, "labels": labels}

    ds = Dataset.from_list([preparar(e) for e in dados])
    print(f"[train] exemplo:\n{ds[0]['input_ids'][:20]}... "
          f"len={len(ds[0]['input_ids'])} tokens_resposta={sum(1 for l in ds[0]['labels'] if l != -100)}")

    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=SAIDA,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = transformers.Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    print("[train] iniciando treinamento...")
    trainer.train()
    model.save_pretrained(SAIDA)
    tokenizer.save_pretrained(SAIDA)
    print(f"[train] ADAPTER SALVO em {SAIDA}")


if __name__ == "__main__":
    main()
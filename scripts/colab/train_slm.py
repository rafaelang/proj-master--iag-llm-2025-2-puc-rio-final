"""v0.5 · Lab — destilação via LoRA do SLM (Qwen2.5-1.5B) no dataset curado.

Spec: "o modelo grande gera o dataset da tarefa; treina-se um SLM".
Roda na T4 (Colab): modelo em 4-bit (bitsandbytes) + LoRA (peft).

Correções da PoC (R5): dataset já CUPADO (dataset_sintetico.json com casos de
abstenção e compostos) — a curadoria é feita no gen_dataset.py ANTES do treino.

Saída: /content/data/processed/slm_adapter/ (adapter + tokenizer)
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
DATASET = os.environ.get("TRAIN_DATASET",
                         "/content/data/golden_set/adaptacao/dataset_sintetico.json")
SAIDA = os.environ.get("TRAIN_SAIDA", "/content/data/processed/slm_adapter")


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
    print(f"[train] dataset curado: {len(dados)} itens "
          f"({sum(1 for d in dados if d.get('tipo') == 'abstencao')} abstenção, "
          f"{sum(1 for d in dados if d.get('tipo') == 'composta')} composta)")

    def fmt(ex):
        return {"text": f"Pergunta: {ex['pergunta']}\nResposta: {ex['resposta']}"}

    ds = Dataset.from_list([fmt(e) for e in dados])

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=512, padding=False)

    ds = ds.map(tokenize, remove_columns=["text"])

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
        num_train_epochs=3,
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

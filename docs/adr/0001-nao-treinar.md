# Não treinar: RAG + frontier é a produção

## Status

accepted

## Contexto e decisão

A disciplina pede comparar prompt, RAG, fine-tuning e pré-treino. Medimos as quatro
abordagens e **decidimos não treinar**: **RAG + LLM frontier** é a arquitetura de
produção, com um **SLM local** apenas na rota simples (custo zero).

## Considered Options

- **LoRA destilado no SLM** (v0.5/v0.5b): reduz alucinações, mas **piora a acurácia**
  por super-cautela (0.40 → 0.30); não bate o baseline.
- **Fine-tune do Laya como roteador** (v0.7): melhora in-receita (held-out 0.99/1.00)
  mas **colapsa na régua congelada** (0.676 < R2 0.941) e **destrói a calibração**
  (ECE 0.052 → 0.337).
- **Kev-0.8B / Laya zero-shot como roteador**: 0.647 — abaixo do cascade (0.912/0.941).

## Consequences

O ganho do projeto veio de **engenharia** (ingestão anti-garbage, OCR de figuras, RRF
ponderado, regra V5), não de treino. A decisão é reconfirmada em três releases
independentes; qualquer tentativa futura de treino precisa bater o cascade **na régua
congelada**, não só in-distribution.

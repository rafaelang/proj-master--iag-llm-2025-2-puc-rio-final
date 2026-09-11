# v0.5 · Adaptação — evidência (comparativo e decisão)

> Gerado por `scripts/avaliadores/avaliar_v5.py` em 2026-09-11T11:23:12.
> Dataset: `data/golden_set/rag/perguntas.json` (20 perguntas, congelado).
> Gerador frontier: `deepseek-chat` · Juiz: `deepseek-v4-pro`.

## 1. Mini-experimento: prompt-only × RAG (acurácia via juiz — R3)

| Métrica | Prompt-only (sem evidência) | RAG (com evidência) |
|---|---|---|
| Abstenção correta | 0.850 | 0.800 |
| **Acurácia end-to-end** (juiz) | **0.833** | **0.737** |
| Nota média (1–5) | 4.39 | 3.84 |
| Alucinações | 2 | 0 |
| Citação presente | 0.000 | 1.000 |

> **Leitura honesta:** o requisito da spec é **citação + abstenção** (grounding), não acurácia bruta. O prompt-only (frontier sem evidência) até acerta mais no corpus (modelo forte memorizou o tema), mas tem **0% de citação e 2 alucinações fora do corpus** (#18/#20 respondem em vez de abster); o RAG tem **0 alucinações e 100% de citação** — cumpre o requisito que a spec pede. A métrica agora é acurácia via juiz (R3), corrigindo a falha metodológica da PoC.

## 2. SLM (base × destilado) no RAG — matriz 2×2 e juiz de correção

| Sujeito | RAG | Abstenção correta | Acurácia (juiz) | Nota | Alucinações |
|---|---|---|---|---|---|
| base | sim | 0.800 | 0.400 | 2.85 | 5 |
| base_no_rag | não | 0.700 | 0.350 | 2.6 | 6 |
| destilado | sim | 0.650 | 0.300 | 2.35 | 4 |
| destilado_no_rag | não | 0.600 | 0.450 | 2.8 | 5 |

## 3. Decisão (resumida)

A decisão completa (com checklist go/no-go, anti-gatilhos e TCO) está em `docs/v05.md`. Síntese dos resultados: **não vale treinar/pré-treinar** — a destilação (LoRA curado, 29 itens) reduziu alucinações (5→4) mas piorou a acurácia (0.40→0.30) por **super-cautela** (abstém mesmo com evidência em #1/#3); o RAG (frontier) permanece a escolha de produção: 0 alucinações + 100% de citação, atendendo o requisito de grounding da spec.


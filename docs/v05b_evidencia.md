# v0.5b · Expansão do corpus — evidência (comparativo v0.5 × v0.5b)

> Gerado por `scripts/avaliadores/avaliar_v5b.py` em 2026-09-11T16:35:07.
> Dataset: `data/golden_set/rag/perguntas_v05b.json` (20 congeladas + novas, v0.5b).
> Baseline v0.5: corpus 10 docs / 272 chunks · v0.5b: corpus expandido.

## 1. Mini-experimento: prompt-only × RAG — corpus expandido

| Métrica | Prompt-only | RAG |
|---|---|---|
| Abstenção correta | 0.909 | 0.841 |
| **Acurácia end-to-end** (juiz) | **0.884** | **0.705** |
| Nota média (1–5) | 4.49 | 3.86 |
| Alucinações | 3 | 2 |
| Citação presente | 0.000 | 1.000 |

## 2. Comparativo v0.5 × v0.5b (mesmas perguntas congeladas + novas)

| Métrica | v0.5 RAG | v0.5b RAG | delta |
|---|---|---|---|
| Acurácia end-to-end | 0.737 | 0.705 | -0.032 |
| Alucinações | 0 | 2 | - |
| Citação presente | 1.0 | 1.000 | - |

> O executável local do RAG (v0.5b) responde com o corpus expandido (PDF/MD/IPYNB/PPTX). A comparação direta com a v0.5 é limitada porque as perguntas novas (21–44) só existem no corpus expandido; a coluna `v0.5` usa somente as 20 congeladas.

## 3. SLM (base × destilado) — matriz 2×2 no corpus expandido

| Gerador | Sem RAG (memória) | Com RAG (grounding) |
|---|---|---|
| **SLM base** (Qwen2.5-1.5B) | abstenção 0.818 · **acurácia 0.364** · aluc. 13 | abstenção 0.727 · **acurácia 0.295** · aluc. 8 |
| **SLM destilado** (LoRA curado) | abstenção 0.818 · **acurácia 0.386** · aluc. 14 | abstenção 0.682 · **acurácia 0.364** · aluc. 9 |

> **Leitura honesta v0.5b:** no corpus expandido (44 perguntas) o SLM permanece abaixo do frontier RAG/acurácia (0.295–0.386 vs 0.705). A destilação não manteve o ganho de grounding da v0.5 (alucinações: com RAG base 8 × destilado 9; sem RAG 13 × 14). A decisão da v0.5 se **mantém**: RAG (frontier) é a escolha de produção; o SLM segue como executor operacional da v0.4.


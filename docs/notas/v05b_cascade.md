# v0.5b · P2 — Fluxo completo multiagente (roteador cascade) nas 44Q

> Avaliação do fluxo **cascade de produção** nas 44 perguntas do golden v0.5b
> **corrigido** (P1), com o RAG de P1 (índice texto, pool 60):
> SIMPLES → **SLM local** (Qwen2.5-1.5B GGUF), COMPLEXA → **frontier pro**
> (deepseek-v4-pro), fallback cruzado sob falha. `AGENTE_ROTEADOR=cascade`
> (limiar 0.60), `AGENTE_SIMPLES=slm`, `AGENTE_COMPLEXA=pro`.
>
> Responde à **R8 não medida** (slide 48 — tráfego/custo por rota) e à lacuna da
> análise: o fluxo completo não havia sido avaliado nas 44Q. Julgamento por rota
> com o juiz R3 (`avaliar_v5`), mesmo prompt das releases.
>
> Evidência crua: `data/processed/v05b_ab/cascade_v05b_p2.jsonl` (+ `_resumo.json`).

## 1. Roteamento (cascade, limiar 0.60)

| Rota | n | Gerador | Per estrato (n SIMPLES/COMPLEXA) |
|---|---|---|---|
| simples | 34 | SLM local (US$ 0) | rotineira 22/3 · composta 4/5 · negativa 3/1 · adversarial 5/1 |
| complexa | 10 | pro (API) | 2 perguntas caíram p/ SLM via fallback (pro falhou/empty) |

Geradores usados de fato: **SLM 36** × **pro 8**. Fallbacks: 2 (rota complexa → SLM).

## 2. Resultados por rota (juiz R3)

| Rota | n | Acurácia | Abstenção correta | Alucinações | Nota | Citação | Custo US$ | Tokens API |
|---|---|---|---|---|---|---|---|---|
| **simples (SLM)** | 34 | **0.606** (20/33) | 0.882 | **8** | 3.64 | 1.000 | **0.000** | 0 / 0 |
| **complexa (pro)** | 10 | **0.800** (8/10) | 0.900 | 0 | 4.20 | 1.000 | 0.012 | 11.587 / 8.116 |
| **TOTAL** | 44 | **0.651** (28/43) | 0.886 | 8 | 3.77 | 1.000 | **0.012** | 11.587 / 8.116 |

## 3. Leitura — o SLM na rota real (R8)

O SLM julgado **apenas nas 34 perguntas que a cascata envia para ele**:

| Métrica | v0.5b (SLM nas 44Q, pessimista) | **P2 (SLM só rota simples, 34Q)** |
|---|---|---|
| Acurácia (destilado+RAG) | 0.364 | **0.606** (+0.24) |
| Acurácia (base+RAG) | 0.295 | **0.606** (+0.31) |
| Alucinações | 9 | 8 |

O pessimismo artificial da v0.5b é **confirmado e quantificado**: o SLM roda as
perguntas que a cascata realmente lhe envia (factuais/rotineiras) com acurácia
0.606 — muito acima do 0.364/0.295 medido quando se julgavam as 44 (incluindo as
complexas que em produção vão ao frontier).

**O elo fraco do SLM segue sendo a alucinação** (8 nas rotineiras): erra
respondendo com fato inventado (#02/#05/#24/#29/#34/#35/#42/#43) e falha em
abster nos adversarial (#43 respondeu fora do corpus). Não é super-cautela — é o
comportamento de "memória sem grounding" que o P3 ataca (prompt RAG da rota simples).

## 4. Custo × acurácia (R7 — decisão de produção)

| Sistema | Acurácia (44Q) | Custo API / 44Q |
|---|---|---|
| **Cascade (SLM rota simples + pro complexa)** | **0.651** | **US$ 0.012** |
| Frontier RAG-only (P1, pro em todas) | **0.886** | US$ 0.066 (estimado) |

A cascade custa ~**5,5× menos** (US$ 0.012 vs 0.066) mas perde 0.235 de acurácia.
O trade-off é ditado pela rota simples: 34/44 perguntas custam US$ 0 (SLM local),
porém com acurácia 0.606 e 8 alucinações. A decisão "não treinar para volume de
startup" ganha a régua de custo: **se a acurácia importa, frontier-only (0.886);
se o custo importa, cascade (0.651)**. O gap da rota simples (alucinação) é o
próximo alvo (P3), não o roteador.

## 5. Por estrato (cascade total)

| Estrato | n | acertos | aluc | abstenção ok |
|---|---|---|---|---|
| rotineira | 25 | 15/25 | 6 | 24/25 |
| composta | 9 | 8/9 | 0 | 8/9 |
| negativa | 4 | 2/4 | 1 | 2/4 |
| adversarial | 6 | 5/6 | 1 | 5/6 |

## 6. Metodologia e limites

- **Ambiente:** o experimento é CPU/API-bound (SLM GGUF local + API DeepSeek). As
  sessões Colab provaram-se instáveis para esta carga (6 perdas em ~30 min de uso,
  cada uma exigindo recompilar llama-cpp-python ~20 min); o AGENTS.md limita o SLM
  local ao ambiente local/CLI. O experimento rodou **local** (venv, mesmo código/
  modelos), com salvamento incremental por pergunta; a medição é reproduzível via
  `RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_cascade_v05b.py`.
- Juiz: 1 pergunta (#08) sem veredito (parse falhou); denominador 43.
- Fallbacks: 2 (pro retornou vazio → SLM); o custo de API reflete as 8 gerações pro.
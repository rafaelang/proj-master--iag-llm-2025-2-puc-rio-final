# v0.6 · Cenário de teste — Kev como roteador SIMPLES/COMPLEXA (isolado)

> Data: 2026-09-28 · Endpoint **Kev-0.8B** (`jaredpalmer/kev-0.8b`, Qwen3.5-0.8B-Base,
> LoRA r=16, temp calibrada 2.35) servido em **Colab T4** (torch, bf16) ·
> Dataset do estudo do roteador `data/processed/rag_v4/roteador_rotulos.json`
> (20 canônicas + 14 typos, **artefato, não golden set**) · Comparativo com os
> backends medidos em `avaliar_roteador.py` (`avaliacao_roteador.json`).

## 1. O que foi testado

Cenário **isolado do roteador**: Kev classifica cada pergunta em SIMPLES/COMPLEXA
via `POST /v1/systemone` (API TypeSafe/System One), com duas variantes de questão:

| Variante | Formato | Decisão |
|---|---|---|
| `choice` | critérios `simples`/`complexa` (mesmo critério dos roteadores atuais) | rótulo direto + `probabilities` |
| `noul` | "A pergunta é COMPLEXA?" | `noul ≥ 0.5` → COMPLEXA |

Métricas idênticas a `avaliar_roteador.py` (acurácia canônicas/typos/total,
falso-S/falso-C, latência) + extras Kev: **confiança média (p_max)** e **varredura
de limiar** (decide sozinho se p_max ≥ limiar; senão escala — conceito cascade).

## 2. Resultados (34 itens)

| backend | acc canônicas | acc typos | acc total | falso-S | falso-C | lat mediana (s) |
|---|---|---|---|---|---|---|
| tfidf (R2) | 1.000 | 0.857 | **0.941** | 2 | 0 | 0.0015 |
| slm (R1) | 0.900 | 0.929 | **0.912** | 1 | 2 | 5.718 |
| flash (API) | 0.700 | 0.714 | 0.706 | 10 | 0 | 1.283 |
| **kev-choice** | 0.750 | 0.500 | 0.647 | 0 | 12 | 0.132 |
| **kev-noul** | 0.400 | 0.357 | 0.382 | 0 | 21 | 0.116 |

Confiança (p_max): choice média 0.625 (min 0.50 / max 0.81) · noul média 0.668
(min 0.50 / max 0.82). Zero exceções. Custo: US$ 0 local (GPU Colab).

### 2.1 Varredura de limiar (Kev-choice) — conceito cascade

| limiar | n auto | taxa auto | acurácia auto | n escala |
|---|---|---|---|---|
| 0.50 | 34 | 1.000 | 0.647 | 0 |
| **0.60** | 21 | 0.618 | **0.810** | 13 |
| 0.70 | 5 | 0.147 | 1.000 | 29 |
| 0.80 | 2 | 0.059 | 1.000 | 32 |
| 0.90 | 0 | 0.000 | – | 34 |

## 3. Leitura

- **Kev-0.8B zero-shot NÃO é competitivo** com os roteadores atuais: fica abaixo
  até do flash (0.706) e muito abaixo do R2 (0.941) / R1 (0.912). As duas variantes
  erraram 100% das rotineiras factuais do tipo "O que é X?" → rotearam para
  COMPLEXA (falso-C 12 e 21; **falso-S = 0**).
- **Direção do erro é segura mas cara**: nunca deixa passar uma COMPLEXA para a
  rota simples (falso-S 0), porém sobre-roda SIMPLES→COMPLEXA, que na cascata de
  produção significa **escalar para o pro (API paga)** — o oposto do objetivo de
  custo do roteador (rota simples = SLM US$ 0).
- **`choice` ≫ `noul`** (0.647 × 0.382): a pergunta com opções explícitas segura a
  calibração; a binária noul afoga em COMPLEXA. Confiança global baixa (0.5–0.8)
  — nenhuma decisão auto confiante no regime útil (0.90 não decide nada).
- **Typos derrubam Kev** (choice 0.75 → 0.50): sem n-grams de caractere, o modelo
  perde robustez a erro de ASR que o R2 absorve por design.
- **Latência** (0.13 s) melhor que slm (5.7 s) e flash (1.28 s), mas exige GPU
  dedicada e fica longe do R2 (< 2 ms, CPU). Nota: sem `flash-linear-attention`/
  `causal_conv1d` (kernels opcionais) o servidor caiu em implementação de
  referência (mais lenta; medição ainda válida para o cenário).

## 4. Decisão do cenário

**Não adotar Kev-0.8B zero-shot como roteador.** A alavanca que o Kev oferece —
fine-tune no seu domínio — não se paga aqui: o dataset de roteamento tem 20
canônicas (34 com typos), e o próprio repo de Kev reporta que ganhos mensuráveis
exigem ~400 exemplos. O R2 (TF-IDF+XGB, cascade@0.60) já entrega 0.941 a < 2 ms e
US$ 0 sem GPU. Cenário encerra como **contra-evidência documentada** (baseline não
batido), no mesmo espírito do P3 (prompt/LoRA do SLM) e do estudo do limiar
(`docs/v06_rotas_estratos.md`).

## 5. Reprodução

- **Endpoint (na VM Colab):** `python scripts/colab/servir_kev.py` (env `KEV_RUN`,
  `KEV_PORT`, `KEV_TIMEOUT_S`) — clona `jaredpalmer/kev`, `uv sync --extra serve`
  e deixa `kev.serve` em background (127.0.0.1:8009).
- **Cenário:** `KEV_BASE_URL=http://127.0.0.1:8009 python scripts/avaliadores/avaliar_roteador_kev.py`
  (`--questao choice|noul`, `--dataset`, `--saida-json*`). Auto-contido
  (`urllib` stdlib, `_metricas` copiado de `avaliar_roteador.py`).
- **Dados (fora do Git):** `data/processed/rag_v4/avaliacao_roteador_kev_{choice,noul}.{json,jsonl}`.
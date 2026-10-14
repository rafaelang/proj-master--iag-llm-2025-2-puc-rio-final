# v0.7 — Evidência medida (golden expandido 516Q; apêndice pós-v1.0)

> Todos os números foram medidos **no mesmo golden** (`data/golden_set/rag/perguntas_v07.json`,
> 516Q = 132 congeladas da v0.6 + 384 novas) sobre o índice de produção. SLM = Qwen2.5-1.5B
> GGUF local; pro = DeepSeek; juiz = DeepSeek (rotulagem congelada). JSONs em
> `data/processed/` são dado intermediário; a evidência oficial é este documento.
> Detalhes de processo em `docs/v07.md`; cenários de roteador em `docs/notas/`.

## 1. P1 isolado — retrieval (516Q)

`data/processed/rag/retrieval_v07.json` · RRF BM25 1.0 × denso 1.5, sem rerank, pool 60, top-5.

| estrato | n | recall@5 | MRR |
|---|---|---|---|
| rotineira | 266 | 0.812 | 0.645 |
| composta | 109 | 0.881 | 0.724 |
| negativa | 66 | 0.803 | 0.648 |
| adversarial | 75 | — (sem docs_esperados) | — |
| **geral (441 com docs)** | 441 | **0.828** | **0.665** |

Comparação com a régua histórica: v0.5b (44Q) recall@5 0.763 · v0.6 (132Q) 0.659 DOC.
Com 4× mais casos que a v0.6, o recall@5 **sobe para 0.828** (mesma configuração de
produção) — o retriever mantém o perfil e a amostra maior **não** confirma o teto
suspeitado; o alvo de página segue sendo o limitante (ver v0.6).

## 2. P2 fim-a-fim — cascade roteador→gerador→juiz (516Q)

`data/processed/v05b_ab/cascade_v07_p2.jsonl` (+ `_resumo.json`). 489/516 julgadas.

| estrato | n | julgadas | acertos | acurácia | alucinações |
|---|---|---|---|---|---|
| rotineira | 266 | 250 | 166 | 0.664 | 37 |
| composta | 109 | 98 | 62 | 0.633 | 20 |
| negativa | 66 | 66 | 6 | 0.091 | 1 |
| adversarial | 75 | 75 | 64 | 0.853 | 11 |
| **total** | **516** | **489** | **298** | **0.609** | **69** |

Rotas: **simples (SLM) 459** = 0.587 (aluc 68) · **complexa (pro) 57** = 0.792 (aluc 1).
Abstenção correta 0.773 · citação 0.997 · custo **US$ 0.07369** (**0.000143/Q**) ·
latência média 62,6 s.

## 3. Frontier RAG (baseline, pro em todas as perguntas)

`data/processed/v07_ab/frontier_v07.jsonl` (+ `_resumo.json`). 511/516 julgadas.

| estrato | n | julgadas | acertos | acurácia | alucinações |
|---|---|---|---|---|---|
| rotineira | 266 | 263 | 205 | 0.779 | 6 |
| composta | 109 | 109 | 88 | 0.807 | 2 |
| negativa | 66 | 64 | 57 | 0.891 | 2 |
| adversarial | 75 | 75 | 75 | 1.000 | 0 |
| **total** | **516** | **511** | **425** | **0.832** | **10** |

Abstenção correta 0.878 · citação 0.997 · custo **US$ 0.5603** (**0.001086/Q**) ·
latência média 10,0 s.

**Leitura central:** no golden 516Q o frontier mantém **+0,22 de acurácia** (0.832 vs
0.609) por **~7,6× o custo** (0.001086 vs 0.000143 US$/Q). O elo fraco continua a
**negativa** na rota simples (0.091) — a regra determinística V5 (projeção 0.786)
segue como a melhoria de maior retorno e custo zero.

## 4. Comparativo por estágio (amostras crescentes)

| Release | n | retrieval recall@5 | cascade acurácia | cascade US$/Q | frontier acurácia |
|---|---|---|---|---|---|
| v0.5b | 44 | 0.763 | 0.651 | 0.00027 | — |
| v0.6 | 132 | 0.659 (DOC) | 0.575 | 0.00012 | — |
| **v0.7** | **516** | **0.828** | **0.609** | **0.000143** | **0.832** |

## 5. Cenários de roteador — contra-evidência (mesma régua de 34 itens)

Artefato de cenário `data/processed/rag_v4/roteador_rotulos.json` (20 canônicas + 14 typos),
**não** golden. Baselines de produção: R2 (TF-IDF+XGB) e R1 (SLM few-shot).

| backend | acc total | falso-S | falso-C | ECE 10 bins |
|---|---|---|---|---|
| tfidf (R2, produção) | **0.941** | 2 | 0 | — |
| slm (R1, produção) | **0.912** | 1 | 2 | — |
| flash (API) | 0.706 | 10 | 0 | — |
| laya-ft choice (fine-tuned) | 0.676 | 11 | 0 | 0.337 |
| kev-choice (0.8B) | 0.647 | 0 | 12 | — |
| laya multilingual choice (zero-shot) | 0.647 | 9 | 3 | **0.052** |
| kev-noul | 0.382 | 0 | 21 | — |

**Decisão (mantida):** nenhum candidato bate o cascade `R1/R2`; o fine-tune do Laya
melhora in-receita (held-out 0.99/1.00) mas **colapsa na régua congelada** e destrói a
calibração (ECE 0.052 → 0.337). Evidência detalhada em `docs/notas/v06_roteador_kev.md`,
`docs/notas/v06_roteador_laya.md` e `docs/notas/v07_roteador_laya_ft.md`.

## 6. TCO (golden 516Q)

Custo medido (`uso` real da API, preços DeepSeek diretos): cascade **US$ 0.000143/Q**
(~**US$ 0.14/mil**) vs frontier **US$ 0.001086/Q** (~US$ 1.09/mil) — **~7,6× mais barato**.
Roteamento real 89% simples (SLM, US$ 0) / 11% complexa. Detalhes e camada OpenCode Go em
`docs/notas/v07_tco.md`.

## 7. Dados (fora do Git)

- `data/golden_set/rag/perguntas_v07.json` (golden, **entra no Git**)
- `data/processed/rag/retrieval_v07.json`
- `data/processed/v05b_ab/cascade_v07_p2.jsonl` (+ `_resumo.json`)
- `data/processed/v07_ab/frontier_v07.jsonl` (+ `_resumo.json`)
- `data/processed/rag_v4/avaliacao_roteador_*` (cenários de roteador)

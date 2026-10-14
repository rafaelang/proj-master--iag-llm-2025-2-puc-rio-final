# v0.7 · Cenário de teste — Fine-tune do Laya como roteador (`laya-ft`)

> Data: 2026-10-07 · `laya` **0.3.29** (Apache 2.0) · base `multilingual`
> (mmBERT-base ~322M) · **Colab T4** (torch cu130) · treino com o `laya-train`
> oficial (RLCD, 4 epochs, micro-batch 8 / grad-accum 8, lr encoder 2.5e-5 /
> head 1e-4) · servido via `serve.py` custom com checkpoint local registrado.
>
> **Pergunta do cenário:** a alavanca que a v0.6 anotou como intocada — "o
> upstream só reporta ganho com centenas de exemplos rotulados" — com um
> **dataset sintético de treino** dirigido à mesma tarefa de roteamento. Golden
> **intocado** (`data/golden_set/` só leitura para dedup): o treino é dataset
> novo, fora da régua de avaliação.

## 1. Dataset de treino (fora do golden)

Receita do P5/P7 aplicada **fora** do golden (`scripts/colab/gen_rotulos_laya_ft.py`):

1. **Geração** com o pro (`deepseek-v4-pro` via gateway zen), ancorada em chunks
   reais do corpus: SIMPLES = 1 chunk, COMPLEXA = 2 chunk/docs;
2. **Auditoria independente** com o pro: o auditor vê **apenas a pergunta** e aplica
   o critério congelado (`criterio_rotulo.md`) — só entra se `auditor == alvo`;
   concordância 0.975 (513 aprovados / 13 discordâncias);
3. **Dedup** (normalizada exata + Jaccard > 0.80) contra TUDO que é régua de
   avaliação: golden v0.7 (516), v0.6, v0.5b, v0.2 (20) e artefato `roteador_rotulos.json`;
4. **Split** fixo: 400 treino / 100 held-out, seed 42; augmentação de typo (ASR
   pt-BR) em 20% do treino → 462 linhas JSONL no formato do upstream
   (`{"state","questions","gold"}`, choice + noul).

| metrica | valor |
|---|---|
| sobreviventes únicos | 500 (292 SIMPLES / 208 COMPLEXA) |
| chamadas de geração (rodada final) | 407 · candidatos inválidos/vazios 1 |
| concordância auditor independente | 0.975 |
| split | 400 treino / 100 teste |
| linhas JSONL (com typos) | 462 · 62 estados com typo |

Golden congelado; regra de rótulo congelada; nenhum item da régua no treino.
(Obstáculo de infra registrado: o gateway esvaziou por períodos na madrugada;
o gerador ganhou retry/`timeout` no cliente e um fallback via `opencode run`
com o modelo `big-pickle` — o fallback **não foi acionado**, o pro se recuperou.)

## 2. Resultados

### 2.1 Held-out (100, mesma receita do treino) — o FT aprendeu

| modelo / questão | acc | falso-S | falso-C | ECE | Brier | conf média |
|---|---|---|---|---|---|---|
| **laya-ft choice** | **0.99** | 0 | 1 | 0.013 | 0.009 | — |
| **laya-ft noul** | **1.00** | 0 | 0 | 0.000 | 0.000 | 1.00 |
| multilingual zero-shot choice (controle 0.3.29) | 0.52 | 43 | 5 | 0.177 | 0.288 | — |

No domínio **dentro da receita** o fine-tune aprende de fato: multitask quase
perfeita (choice 54/55 S + 45/46 C; noul 100/100).

### 2.2 Régua congelada (artefato `roteador_rotulos.json`, 34) — não bate ninguém

| backend | acc canônicas | acc typos | acc total | falso-S | falso-C | ECE 10 bins | Brier |
|---|---|---|---|---|---|---|---|
| tfidf (R2) | 1.000 | 0.857 | **0.941** | 2 | 0 | — | — |
| slm (R1) | 0.900 | 0.929 | **0.912** | 1 | 2 | — | — |
| flash (API) | 0.700 | 0.714 | 0.706 | 10 | 0 | — | — |
| **laya-ft choice** | 0.650 | 0.714 | **0.676** | **11** | 0 | **0.337** | 0.329 |
| laya-ft noul | 0.650 | 0.643 | 0.647 | 12 | 0 | 0.353 | 0.353 |
| multilingual choice (zero-shot, 0.3.29) | 0.650 | 0.643 | **0.647** | 9 | 3 | **0.052** | 0.197 |
| kev-choice (0.8B) | 0.750 | 0.500 | 0.647 | 0 | 12 | — | — |
| multilingual noul (zero-shot) | 0.500 | 0.357 | 0.441 | 4 | 15 | 0.310 | 0.359 |
| kev-noul | 0.400 | 0.357 | 0.382 | 0 | 21 | — | — |

Zero exceções; `routing.model` = o solicitado em 100% dos casos
(`routing.reason: "explicit model=..."`, repo `/content/laya_ft`).

**Padrão dos 11 erros do `laya-ft`:** todos COMPLEXA previstos SIMPLES com
`conf = 1.00` — as compostas canônicas "O que é X e como Y…" (`#9–14`) e o item
**fora do corpus** `#19` (relatividade geral). Lista em
`data/processed/rag_v4/avaliacao_roteador_laya_laya-ft_choice.jsonl`.

## 3. Leitura

O número em-distribuição é forte (0.99/1.0), mas a régua congelada conta a história:

1. **Memorizou a receita, não a tarefa.** O FT colapsa a direção do erro para
   o **lado errado**: zero falso-C (nunca "custa caro") mas **11 falso-S** — 11
   complexas mandadas para o SLM local de 1.5B, que na v0.6/v0.7 é o elo que
   **alucina**. O ganho de 0.647→0.676 é comprado trocando o erro calibrado e
   bilateral (9S/3C) do zero-shot por um erro unilateral e **máximo de confiança**.
2. **Quebrou o único ativo real do Laya: a calibração.** ECE sobe de 0.052
   (zero-shot, bins quase perfeitos) para **0.337**; confiança média 0.987 e
   33/34 decisões em `conf 1.00`. Sem distribuição de confiança, o sweep de
   limiar (conceito cascade) vira degenerado — não há o que escalar por fé.
3. **O fora-de-corpus ficou pior, não melhor.** A pergunta de Einstein (sem
   suporte no corpus) sai com `conf 1.00` para SIMPLES, sem qualquer dúvida
   evidenciável. No papel de roteador isso **não é abstenção** — nem o zero-shot
   tinha canal de abstenção, mas não afirmava com 1.00 o que não sabe.
4. **Mesmo resultado do P3 (v0.5b) e da leitura da v0.6:** treinar no sintético
   do pro melhora em-distribuição à receita e degrada na régua congelada — com o
   agravante de destruir a calibração.

## 4. Decisão do cenário

**Não adotar; mantém a decisão "não treinar" (P3/v0.5b) e a régua congelada.**
Nada muda no pipeline: roteador de produção segue o cascade `R1/R2` (0.941/0.912
> tudo medido), golden intocado, zero impacto de release. Este cenário encerra
como **contra-evidência documentada** do fine-tuning do Laya para roteamento no
corpus PT-BR — no mesmo espírito do Kev (`docs/notas/v06_roteador_kev.md`), da v0.6
(`docs/notas/v06_roteador_laya.md`) e do SLM (`docs/notas/v05b_p3_slm_ab.md`).

O ativo da v0.6 — **calibração do zero-shot multilingual** — segue sendo o
único motivo para olhar o Laya de novo, e **numa direção que não depende de
treinar**: usar a confiança calibrada do `choice` zero-shot como *gate* em um
cascade (tema de um cenário futuro, não desta release).

## 5. Reprodução

```bash
# 1. treino no Colab (T4), laya 0.3.29 — JSONL com {state, questions, gold}
laya-train --data roteador_treino_laya.jsonl --base multilingual \
           --out /content/laya_ft --epochs 4 --device cuda
#    workaround de bug do pacote: model.to(device) acontece só dentro do
#    train_model, depois do calibration_records pre-treino -> monkeypatch
#    load_checkpoint para mover o modelo para cuda na carga (ver gene dl).

# 2. servir (scripts/colab/servir_laya_ft.py): registra o checkpoint local como
#    "laya-ft" + preloads "multilingual" (controle da mesma versao) e passa o
#    campo model do cliente adiante (o _resolve_model do pacote rejeita nomes
#    locais e cairia no english nao-carregado).

# 3. baterias (mesmo avaliador da v0.6, --saida-json para o held-out)
python scripts/avaliadores/avaliar_roteador_laya.py --modelo laya-ft --questao choice
python scripts/avaliadores/avaliar_roteador_laya.py --modelo laya-ft --questao noul
python scripts/avaliadores/avaliar_roteador_laya.py --modelo multilingual --questao choice
python scripts/avaliadores/avaliar_roteador_laya.py --modelo multilingual --questao noul
python .../avaliar_roteador_laya.py --modelo laya-ft --questao choice --dataset roteador_treino_laya_teste.json \
       --saida-json ..._treino.json --saida-jsonl ..._treino.jsonl
```

- **Dados (fora do Git):** `data/processed/rag_v4/roteador_treino_laya.{jsonl,teste,_meta}.json`,
  `avaliacao_roteador_laya_{laya-ft,multilingual}_{choice,noul}.json`,
  `avaliacao_roteador_laya_*_treino.json`; modelo treinado não versionado na VM.
- **Versões** fixadas no teste: `laya==0.3.29`, base revision `7b928d82`
  (controle e FT na mesma sessão/serviço), T4 cuda.
- **Limitações:** (a) o held-out é **in-distribution à mesma receita** (mesmo
  gerador/critério do treino) — a régua externa real é o artefato de 34 e o
  golden, intocados; (b) o treino é desbalanceado 292/208; (c) a régua de 34 é
  artefato de cenário, não régua de release.
# v0.6 · Cenário de teste — Laya (`convaiinnovations/laya`) como roteador e como juiz

> Data: 2026-09-30 · `laya` **0.3.24** (Apache 2.0) · checkpoints `55cf4c4e`
> · **Colab T4** (torch 2.11.0+cu130, CUDA, `cpu_fallbacks=0`) · servido via
> `laya-serve` (`POST /v1/systemone`, **mesmo wire contract do Kev**) · 3
> checkpoints resident: `multilingual` (mmBERT-base ~322M, ctx 1024),
> `typed-decisions` e `english` (ModernBERT-large ~421M).
>
> **Bateria repetida** de `docs/notas/v06_roteador_kev.md` — mesmos datasets, mesmas
> métricas, mesmo código de comparativo — trocando **apenas o motor**: Kev-0.8B
> → Laya. Eixos: (1) roteador SIMPLES/COMPLEXA; (2) juiz (concordância com o
> juiz pro).

## 1. O que foi testado

### Eixo 1 — roteador (34 itens, `roteador_rotulos.json`)

Mesmos critérios, mesmo dataset, mesmas métricas de `avaliar_roteador_kev.py`.
Variantes: `choice` (rótulo direto) e `noul` ("A pergunta é COMPLEXA?",
≥ 0.5 → COMPLEXA), exatamente como no Kev.

O Laya acrescenta um eixo próprio que o Kev não tem: **escolha de checkpoint**
(`english` / `multilingual` / `typed-decisions`). Medimos os três, porque a
documentação recomenda `multilingual` para tráfego não-inglês e este projeto é
PT-BR. Acrescentamos **ECE (10 bins) + Brier**, porque a promessa do Laya é
probabilidade calibrada — sem isso o sweep de limiar não é interpretável.

### Eixo 2 — juiz (80 itens, mesma amostra de `avaliar_kev_juiz.py`)

`noul` "a resposta está CORRETA?" (rubrica v5) + `score` 1-5, sobre o **mesmo**
sample da fronteira v0.7, para comparação direta com a concordância de 0.512 do
Kev. Nota: o `score` do Laya é o **nível ordinal esperado, base 0**
(confirmado de duas formas: `["a","b","c"] → 1.84 ≈ b` no README do upstream, e
o `score` mínimo observado foi 0.01, impossível numa escala base 1), então
`nota = score + 1`, como no Kev. O avaliador recalcula a nota a partir de
`probabilities` e grava os dois valores: **coincidem em 80/80 itens**, o que
audita a conversão na prática.

## 2. Eixo 1 — resultados (roteador)

| backend | acc canônicas | acc typos | acc total | falso-S | falso-C | lat mediana (s) |
|---|---|---|---|---|---|---|
| tfidf (R2) | 1.000 | 0.857 | **0.941** | 2 | 0 | 0.0015 |
| slm (R1) | 0.900 | 0.929 | **0.912** | 1 | 2 | 5.718 |
| flash (API) | 0.700 | 0.714 | 0.706 | 10 | 0 | 1.283 |
| kev-choice | 0.750 | 0.500 | 0.647 | 0 | 12 | 0.132 |
| kev-noul | 0.400 | 0.357 | 0.382 | 0 | 21 | 0.116 |
| **laya-multilingual-choice** | 0.650 | 0.643 | **0.647** | **9** | 3 | **0.031** |
| laya-multilingual-noul | 0.500 | 0.357 | 0.441 | 4 | 15 | 0.035 |
| laya-english-choice | 0.600 | 0.357 | 0.500 | 4 | 13 | 0.038 |
| laya-typed-decisions-choice | 0.450 | 0.429 | 0.441 | 0 | 19 | 0.039 |

Zero exceções em 34×4 chamadas. Checkpoint servido = o explícito em 100% dos
casos (`routing.reason: "explicit model=..."`), sem roteamento automático ambíguo.

### 2.1 Escolha de checkpoint e calibração

| checkpoint | acc total | falso-S | falso-C | **ECE 10 bins** | Brier | conf. média | conf. máx |
|---|---|---|---|---|---|---|---|
| **multilingual** | **0.647** | 9 | 3 | **0.052** | **0.197** | 0.653 | **0.872** |
| english | 0.500 | 4 | 13 | 0.052 | 0.257 | 0.552 | 0.672 |
| typed-decisions | 0.441 | 0 | 19 | 0.141 | 0.280 | 0.567 | 0.649 |

- **`multilingual` é o melhor dos três** (0.647 vs 0.500 vs 0.441), confirmando a
  recomendação da documentação para tráfego PT-BR. É também o único com
  **separação de confiança útil** (máx 0.872): `english` e `typed-decisions` nunca
  passam de 0.67, então nelas o sweep de limiar é degenerado (a 0.70, **zero**
  decisões automáticas).
- **`typed-decisions` é o pior**, apesar de ser o checkpoint "mais parecido" com a
  tarefa. Refuta a hipótese de que proximidade de task ≈ qualidade.
- **A calibração é a parte genuinamente boa**: no `choice` do multilingual,
  ECE 0.052 com curva monótona e quase perfeita —
  0.538→0.462 · 0.656→0.667 · 0.758→0.800 · 0.871→1.000. A confiança **indica**
  a acurácia. Não é a confiança do Kev (0.5–0.8, correlacionada ao erro).
- **Ressalva de reprodutibilidade:** o servidor loga
  `RuntimeWarning: this checkpoint ships invalid temperatures … choice:11 → 0.5.
  Treat confidence from the affected entries as uncalibrated`
  (`laya_serve.log`). Uma entrada de temperatura do checkpoint fica fora de
  [0.5, 5] e é fixada em 0.5 — o próprio upstream marca essa entrada como
  **não calibrada**. O ECE acima cobre 11 `choice`s de 2 opções; a entrada
  afetada (`choice:11`) cai fora da nossa janela de medição, mas o número não
  deve ser lido como calibração total do modelo.
- **`noul` é descalibrado e pior** (ECE 0.310; confiança 0.841 com acurácia 0.25
  no bin 0.8–0.9). Mesma ordem do Kev (`choice` ≫ `noul`), mas aqui o `noul` é
  também o menos **confiável** — enquanto o `choice` é confiável.

### 2.2 Varredura de limiar — Laya `choice` (conceito cascade)

| limiar | n auto | taxa auto | acurácia auto | n escala |
|---|---|---|---|---|
| 0.50 | 34 | 1.000 | 0.647 | 0 |
| **0.60** | 21 | 0.618 | **0.762** | 13 |
| **0.70** | 12 | 0.353 | **0.833** | 22 |
| 0.80 | 2 | 0.059 | 1.000 | 32 |
| 0.90 | 0 | 0.000 | – | 34 |

O sweep do Laya é **melhor que o do Kev** (Kev a 0.70: 5 automáticas / 1.000;
Laya a 0.70: 12 automáticas / 0.833) — e isso é consequência direta da calibração:
a confiança carrega informação. Ainda assim, em nenhum limiar o Laya passa do
R2 (0.941) nem do R1 (0.912), e o limiar que produz 0.833 deixa 65% das perguntas
escalando — ou seja, Little's Law desfavorável.

## 3. Eixo 2 — resultados (juiz, 80 itens)

| | Laya (multilingual) | Kev (0.8B) | **juiz trivial "sempre CORRETA"** |
|---|---|---|---|
| concordância geral | **0.600** | 0.512 | **0.637** |
| adversarial (20) | 1.000 | 0.050 | 1.000 |
| composta (20) | 0.450 | 0.600 | — |
| negativa (15) | 0.533 | 0.600 | — |
| rotineira (25) | 0.440 | 0.760 | — |
| matriz (pro × motor) | TP 48 · **TN 0** · FP 29 · FN 3 | TP 28 · TN 13 · FP 16 · FN 23 | — |
| concordância com conf ≥ 0.7 | 0.613 (75/80) | 0.800 (35/80) | — |
| MAE da nota (1-5) | 2.02 | 1.84 | — |
| ECE da noul | 0.072 | — | — |

**Este é um resultado negativo, e o número bruto engana.** A concordância de 0.600
fica *acima* da do Kev, mas **abaixo da concordância de um juiz degenerado** que
responde sempre "CORRETA" (51/80 = 0.637), porque o pro julga 29/80 como
incorreta. Um "concordância 0.600 > 0.512 do Kev" seria leitura errada.

- **Leniência total como juiz.** O Laya responde "CORRETA" em **77/80 (96%)**;
  discorda do pro apenas 3 vezes. A matriz é degenerada: **TN = 0** (nunca
  concorda que algo está errado) e **FP = 29** (diz que está correto quando o pro
  diz que está errado). Ele não está "certa em 60% das vezes" — está dizendo
  "correto" quase sempre.
- **O 1.000 em adversarial é artefato**, não resultado. Os 20 itens
  adversarial do sample têm `pro_correta = True` nos 20 casos (abstenção
  `NAO_SEI` conta como resposta correta), então um juiz degenerado que sempre
  diz "CORRETA" também faz 1.000 ali. O mesmo vale para o 0.050 do Kev: os dois
  extremos são produto da distribuição do estrato, não da capacidade do motor.
- **Colapso ordinal no score.** A nota mediana do Laya é 1.85, 33/80 no nível 1, e
  o **máximo é 3.83 — nunca emite "5"**, apesar de o pro dar nota 5 em 47/80.
  MAE 2.02 (pior que o Kev 1.84). Isso confirma em dados próprios o aviso do
  upstream de que `score` é a área mais fraca.
- **A confiança é quase inútil aqui:** 75/80 acima de 0.7 (o Laya é
  oversconfiante no papel de juiz) e a concordância entre 0.613 nesse recorte é
  exatamente a trivial. ECE 0.072 é bom *para noul em geral*, mas **a
  probabilidade está bem calibrada de um preditor que erra muito** — calibração
  não substitui acurácia.
- **Como triagem, ainda assim não serve:** o único papel que sobra (sinalizar
  casos de alta confiança para revisão humana) seria exercido por um motor que
  nunca diz "incorreto" → não há o que sinalizar.

## 4. Decisão do cenário

**Não adotar o Laya em nenhuma das duas posições.** Nenhum baseline foi batido:

- **roteador:** 0.647 contra R2 0.941 e R1 0.912 — e abaixo do `flash` (API) em
  custo-benefício, já que o Laya exige GPU para os 31 ms;
- **juiz:** 0.600 contra o trivial 0.637.

Três leituras que valem mais que a média de acurácia:

1. **A direção do erro é o que decide, e aqui ela se inverteu contra nós.** O Kev
   errava para o lado *seguro* (falso-S 0, falso-C 12): mandava SIMPLES para
   COMPLEXA, escalate e paga API — desperdício de custo. O Laya faz o oposto
   (**falso-S 9, falso-C 3**): manda 9 complexas para a rota simples, que é o SLM
   local de 1.5B. Na cascata de produção, complexa errada **não é um custo, é
   alucinação** — a v0.6 mediu composta/negativa como o elo mais fraco e
   alucinação como o elo fraco do SLM. Trocar "caro" por "errado" não é melhoria.
2. **Calibração é real e é o que o Kev não tinha** (ECE 0.052 vs sweep degenerado
   do Kev; bins quase perfeitos). Isso é exatamente a propriedade que a
   arquitetura de cascade precisa para poder **abstencer/escalar por confiança** —
   e é o ativo a explorar em outro lugar, se houver dado rotulado. Só não resolve
   o problema que temos: o problema é o argmax, não a confiança.
3. **Fine-tuning é a alavanca declarada pelo upstream, e aqui não compensa
   (mesma razão do Kev).** O dataset de roteamento tem 34 itens (20 canônicas);
   o próprio upstream só reporta ganho mensurável com centenas de exemplos, e o
   golden do projeto é congelado. Medir ganho de fine-tune exigiria gerar
   rótulos — forbidido pela regra de não re-editar golden para melhorar métrica.

Cenário encerra como **contra-evidência documentada**, no mesmo espírito do Kev
(`docs/notas/v06_roteador_kev.md`), do P3 do SLM (`docs/notas/v05b_p3_slm_ab.md`) e do estudo
do limiar (`docs/notas/v06_rotas_estratos.md`).

## 5. Reprodução

```bash
# 1. endpoint na VM Colab (T4): sobe laya-serve na 8010 com os 3 checkpoints
python scripts/colab/servir_laya.py     # env LAYA_PORT, LAYA_DEVICE, LAYA_MODELS, LAYA_TIMEOUT_S

# 2. eixo 1 — roteador (mesma bateria do Kev)
LAYA_BASE_URL=http://127.0.0.1:8010 python scripts/avaliadores/avaliar_roteador_laya.py \
    --questao choice --modelo multilingual
LAYA_BASE_URL=http://127.0.0.1:8010 python scripts/avaliadores/avaliar_roteador_laya.py \
    --questao noul    --modelo multilingual
# eixo de checkpoint: --modelo english | typed-decisions

# 3. eixo 2 — juiz (mesma amostra de 80 do estudo do Kev)
LAYA_BASE_URL=http://127.0.0.1:8010 python scripts/avaliadores/avaliar_laya_juiz.py \
    --sample data/processed/v07_ab/kev_amostra.json \
    --saida  data/processed/v07_ab/laya_juiz_concordancia.json
```

Os dois avaliadores são auto-contidos (só stdlib + HTTP), como os do Kev, e por
isso rodam na VM sem importar o pacote. Ambientes fixados: `laya==0.3.24`,
checkpoint revision `55cf4c4e`, `LAYA_DEFAULT_MODEL=multilingual`.

- **Dados (fora do Git):** `data/processed/rag_v4/avaliacao_roteador_laya_{multilingual,english,typed-decisions}_{choice,noul}.{json,jsonl}`,
  `data/processed/v07_ab/laya_juiz_concordancia.json`,
  `data/processed/v07_ab/resultados_laya_ambiente.json`.
- **Limitação do estudo:** o dataset do roteador é **artefato, não golden set**
  (20 canônicas + 14 typos); o sample do juiz é uma amostra de 80 da fronteira
  v0.7. Nenhum dos dois é a régua de decisão de release — são cenários de
  hipótese, como o do Kev.
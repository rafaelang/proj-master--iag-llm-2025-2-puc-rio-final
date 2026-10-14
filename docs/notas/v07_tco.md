# v0.7 · TCO e break-even por resposta — fluxo deployado (flash/pro)

> Atualizado: **2026-10-14** (preços oficiais DeepSeek correntes; fluxo **deployado**
> no Space = roteador `flash` · simples `flash` · complexa `pro`). Base nos tokens
> **medidos** no golden v0.7 (516Q) e numa amostra local do fluxo deployado.

---

## 1. Preços oficiais das APIs (por 1M de tokens)

`deepseek-v4.1-flash` e `deepseek-v4-pro` (api.deepseek.com; o gateway OpenCode Go
pratica a mesma tabela):

| Modelo | entrada (off-peak) | entrada (peak) | saída (off-peak) | saída (peak) |
|---|---|---|---|---|
| **`deepseek-v4.1-flash`** | **US$ 0,15** | US$ 0,30 | **US$ 0,60** | US$ 1,20 |
| **`deepseek-v4-pro`** | **US$ 0,66** | US$ 1,32 | **US$ 1,98** | US$ 3,96 |

*Peak = 01–04 e 06–10 UTC, seg–sex; fora disso, off-peak (metade). Cache-hit de entrada
custa uma fração (ex.: flash US$ 0,003/M).*

> IDs correntes: `deepseek-v4.1-flash` (o `deepseek-flash`/`deepseek-v4-flash` foram
> descontinuados em set/2026) e `deepseek-v4-pro`.

---

## 2. Tokens por resposta no fluxo **deployado**

O Space roda **roteador `flash` + rota simples `flash` + rota complexa `pro`** (o SLM
local roda só em ambiente local/CLI). Tokens médios por resposta, medidos:

| Etapa | Modelo | Tokens entrada | Tokens saída |
|---|---|---|---|
| Roteador SIMPLES/COMPLEXA | flash | 266 | 73 |
| Rota simples (89% das perguntas) | flash | 1.383 | 429 |
| Rota complexa (11% das perguntas) | pro | 1.653 | 769 |

*Mix de rotas medido no golden v0.7 (516Q): 459 simples / 57 complexa = **89% / 11%**.
Tokens do `pro` do resumo da cascata v0.7 (n=57); do `flash` e do roteador, de uma
amostra local (n=24).*

---

## 3. Custo por resposta (fluxo deployado)

A preços **off-peak**:

| Componente | Conta | Custo/resposta |
|---|---|---|
| Roteador (flash) | 266×0,15 + 73×0,60 (÷1M) | US$ 0,000084 |
| Rota simples (flash, 89%) | 0,89 × (1.383×0,15 + 429×0,60) | US$ 0,000414 |
| Rota complexa (pro, 11%) | 0,11 × (1.653×0,66 + 769×1,98) | US$ 0,000288 |
| **Total** | | **US$ 0,00079** |

**≈ US$ 0,79 por 1.000 respostas** (off-peak) · **≈ US$ 1,58** (peak).

> **Por que subiu vs o doc antigo:** o TCO anterior assumia a **rota simples grátis**
> (SLM local, US$ 0) → 0,00014/Q. No deploy a rota simples é **paga** (flash), e o
> roteador também é uma chamada de API. O mesmo fluxo com o SLM local na simples
> custaria US$ 0,00029/resposta (só a rota complexa `pro`).

---

## 4. Break-even por resposta entregue (API × infra própria)

**Pergunta:** a partir de qual demanda mensal vale ter **infra própria** (SLM local na
rota simples + roteador local `cascade`) em vez de pagar a API (flash)?

**Custo evitado por resposta** ao self-hostar a parte local (roteador + rota simples):

```
roteador flash  = US$ 0,000084
rota simples    = 0,89 × US$ 0,000465 = US$ 0,000414
economia/resposta = US$ 0,000498
```

**Break-even = custo de infra mensal ÷ 0,000498:**

| Infra/mês (self-host do SLM) | Break-even (respostas/mês) |
|---|---|
| **US$ 30** (CPU dedicada / Space `cpu-upgrade`) | **≈ 60.000** |
| **US$ 70** (CPU maior) | **≈ 141.000** |
| US$ 150 (GPU T4 dedicada — teto) | ≈ 301.000 |

| Demanda mensal | API (flash/pro) | Self-host (SLM) | Vence |
|---|---|---|---|
| 10.000 (acadêmico) | US$ 7,9 | US$ 30 | **API** |
| 60.000 | US$ 47 | US$ 30 | **self-host (CPU)** |
| 140.000 | US$ 111 | US$ 70 | **self-host** |
| 300.000 | US$ 238 | US$ 150 | **self-host** |

---

## 5. Decisão e justificativa

1. **Usar as APIs `flash`/`pro` (não self-host) é a escolha racional na demanda do
   projeto.** O break-even fica em **~60.000 respostas/mês** (CPU dedicada) a
   **~141.000** (CPU maior) — acima da demanda acadêmica real (< 10 mil/mês), onde a
   API custa **~US$ 8/mês** contra **US$ 30+** de infra.
2. **O roteador e a rota simples via `flash` são baratos** (~US$ 0,0005/resposta
   juntos) e evitam manter CPU/GPU dedicada e a latência de um SLM síncrono.
3. **A rota complexa (`pro`) é o grosso do custo** (US$ 0,0026/resposta, ~37% do total
   com 11% do tráfego) — é onde otimizar (cache, roteamento mais seletivo) rende mais.
4. **Acima de ~60–140 mil respostas/mês**, self-hostar o SLM (CPU) passa a compensar;
   acima de ~300 mil, mesmo uma GPU T4 dedicada se paga.

**Custo por mil respostas (resposta à Aula 07):** **US$ 0,79** (deployado, off-peak) ·
US$ 1,58 (peak) · US$ 0,29 (se a rota simples voltasse ao SLM local).

---

## 6. Fontes

- Preços: `api-docs.deepseek.com/quick_start/pricing` (out/2026); gateway OpenCode Go
  (`opencode.ai/zen/go`).
- Tokens do fluxo deployado: medição local (roteador/flash) + `cascade_v07_p2_resumo.json`
  (pro).
- Mix de rotas: `data/processed/v05b_ab/cascade_v07_p2_resumo.json` (516Q).
- Hereditário: `docs/notas/v10_tco_cenarios.md` (3 cenários + infra).

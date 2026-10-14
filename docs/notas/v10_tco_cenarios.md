# v1.0 · TCO e break-even (3 cenários — régua da Aula 07/08)

> Data: 2026-09-17 · Conta de hospedagem/custo do assistente em produção, seguindo a
> régua da Aula 07 (`SLM_em_Produção.pptx`) e da Aula 08 (slide 14 — break-even com os
> quatro números: volume, utilização, preço da API, custo de operação). Base nos custos
> **medidos** na re-medição v1.0 (`docs/notas/v10_remediacao.md`) e na v1.0
> (`docs/v10_evidencia.md`).

---

## 1. Custos medidos (base da conta)

Do cascade na re-medição (132Q, `docs/notas/v10_remediacao.md` §3):

| Item | Valor |
|---|---|
| Custo/consulta (cascade, medido) | **US$ 0.000144** |
| Custo/1.000 consultas (cascade) | **US$ 0.14** |
| Custo/consulta (pro forçado, teto) | US$ 0.0015 |
| Custo/1.000 consultas (pro forçado) | US$ 1.50 |
| Tokens médios por consulta complexa | ~2.166 (2.299 prompt + 1.168 completion /16Q) |
| Mix de rotas (132Q) | 88% simples (SLM local, US$ 0) · 12% complexa (pro) |
| Preços DeepSeek (oficiais, out/2026) | `deepseek-v4.1-flash`: US$ 0,15/M in · US$ 0,60/M out · `deepseek-v4-pro`: US$ 0,66/M in · US$ 1,98/M out (off-peak) |
| SLM local (Qwen 1.5B GGUF) | US$ 0.00 por chamada (roda em CPU/GPU própria) |

> A rota **simples custa US$ 0** (SLM local) — o custo total é dominado pela rota
> complexa (12% das consultas). Por isso o cascade (0.000144/Q) é ~10× mais barato que
> o frontier forçado (0.0015/Q).

---

## 2. Cenário 1 — Startup (< 1M tokens/mês)

| Item | Valor |
|---|---|
| Volume | 5.000 consultas/mês ≈ 150k tokens/mês (mix real) |
| Custo API (cascade) | 5.000 × US$ 0.000144 = **US$ 0.72/mês** |
| Custo API (pro forçado) | 5.000 × US$ 0.0015 = US$ 7.50/mês |
| Operação (HF Space free/CPU) | US$ 0 |
| **TCO mensal** | **US$ 0.72** (cascade) |

**Decisão (Aula 08, slide 22 — anti-gatilhos):** volume < 1M tok/mês → **usar API**,
não treinar, não self-host. O cascade com SLM local já é gratuito na rota simples.

---

## 3. Cenário 2 — Média empresa (1–10M tokens/mês)

| Item | Valor |
|---|---|
| Volume | 50.000 consultas/mês ≈ 1.5M tokens/mês |
| Custo API (cascade) | 50.000 × US$ 0.000144 = **US$ 7.20/mês** |
| Custo API (pro forçado) | 50.000 × US$ 0.0015 = US$ 75/mês |
| Operação (Colab T4 pago / HF cpu-upgrade) | ~US$ 25–30/mês |
| **TCO mensal** | **~US$ 32–37** (cascade + infra) |

**Decisão:** o cascade + SLM local continua vencendo; a infra (Space `cpu-upgrade`
~US$ 30/mês ou Colab Pro) domina o custo. Treinar só se a **forma** falhar (dado do P3
v0.5: LoRA não bateu baseline → não treinar).

---

## 4. Cenário 3 — Grande volume / regulado (> 100M tokens/mês)

| Item | Valor |
|---|---|
| Volume | 2M consultas/mês ≈ 60M tokens/mês |
| Custo API (cascade) | 2M × US$ 0.000144 = **US$ 288/mês** |
| Custo API (pro forçado) | 2M × US$ 0.0015 = US$ 3.000/mês |
| Self-host SLM (1× GPU T4 ~US$ 150/mês ou Colab) | substitui a rota simples (88%) |
| Operação (GPU + MLOps) | ~US$ 150–250/mês |
| **TCO mensal (cascade self-host parcial)** | **~US$ 300–450** |

**Decisão:** no volume alto, **self-host do SLM (rota simples) se paga** — elimina a
dependência de API nas 88% simples. O frontier (pro) permanece só como escalada (12%).
Regulado (LGPD): o dado "não sai do perímetro" — self-host é requisito de fluxo, não
só de custo (Aula 07, slide "Brasil").

---

## 5. Break-even por resposta (atualizado — fluxo deployado flash/pro)

> **Atualização 2026-10-14.** O cálculo original desta seção assumia a rota simples
> **grátis** (SLM local) e dava break-even de ~1,1M consultas/mês. No fluxo **deployado**
> (roteador `flash` · simples `flash` · complexa `pro`) a rota simples é **paga** e o
> roteador também é uma chamada de API, então o break-even **cai para ~60–140 mil
> respostas/mês**. Conta completa, tokens medidos e preços atuais em
> `docs/notas/v07_tco.md`.

**Economia por resposta** ao self-hostar a parte local (roteador + rota simples) =
US$ 0,000084 (roteador) + 0,89 × US$ 0,000465 (simples) = **US$ 0,000498/resposta**.

| Infra/mês (self-host do SLM) | Break-even (respostas/mês) |
|---|---|
| **US$ 30** (CPU dedicada / Space `cpu-upgrade`) | **≈ 60.000** |
| **US$ 70** (CPU maior) | **≈ 141.000** |
| US$ 150 (GPU T4 dedicada — teto) | ≈ 301.000 |

| Demanda total/mês | API (flash/pro) | Self-host (SLM) | Vence |
|---|---|---|---|
| 10.000 (acadêmico) | ~US$ 8 | US$ 30 | **API** |
| 60.000 | ~US$ 47 | US$ 30 | **self-host (CPU)** |
| 140.000 | ~US$ 111 | US$ 70 | **self-host** |
| 300.000 | ~US$ 238 | US$ 150 | **self-host** |

**Leitura:** na demanda do projeto (< 10 mil/mês) a **API vence** — é a justificativa
para usar `flash`/`pro` em vez de manter infra própria. O self-host do SLM só compensa a
partir de **~60–140 mil respostas/mês** (CPU) ou **~300 mil** (GPU dedicada).

---

## 6. Decisão consolidada

1. **Manter o cascade (roteador local + SLM local na simples + pro na complexa)** —
   US$ 0.000144/consulta, ~10× mais barato que frontier forçado.
2. **Não treinar/não self-host dedicado** — volume acadêmico < 1M tok/mês; o SLM local
   já roda de graça; LoRA/DPO refutados com número (P3 v0.5).
3. **Infra de produção atual (HF Space cpu-upgrade + dataset privado) é suficiente** —
   US$ 0 na rota simples; o único custo recorrente é o Space (~US$ 0 no free tier /
   ~US$ 30 com cpu-upgrade) e a API complexa (~12% das consultas).

**Custo por mil consultas (resposta da Aula 07):** **US$ 0.14** (cascade, medido na
re-medição) · **US$ 1.50** (pro forçado, teto).

---

## 7. Fontes

- Custos medidos: `docs/notas/v10_remediacao.md` §3 (re-medição 132Q) e
  `docs/v10_evidencia.md` §5 (v1.0).
- Preços DeepSeek: `scripts/avaliadores/avaliar_cascade_v05b.py` (PRECO_IN/OUT).
- Régua: `AULAS/PROJ/aula07/SLM_em_Produção.pptx` (TCO, 3 cenários, Brasil) e
  `AULAS/PROJ/aula08/slm_usar_treinar_alinhar.pptx` (slide 14 — break-even).
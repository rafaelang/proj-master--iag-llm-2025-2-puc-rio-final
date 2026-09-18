# v1.0 · TCO e break-even (3 cenários — régua da Aula 07/08)

> Data: 2026-09-17 · Conta de hospedagem/custo do assistente em produção, seguindo a
> régua da Aula 07 (`SLM_em_Produção.pptx`) e da Aula 08 (slide 14 — break-even com os
> quatro números: volume, utilização, preço da API, custo de operação). Base nos custos
> **medidos** na re-medição v1.0 (`docs/v10_remediacao.md`) e na v1.0
> (`docs/v10_evidencia.md`).

---

## 1. Custos medidos (base da conta)

Do cascade na re-medição (132Q, `docs/v10_remediacao.md` §3):

| Item | Valor |
|---|---|
| Custo/consulta (cascade, medido) | **US$ 0.000144** |
| Custo/1.000 consultas (cascade) | **US$ 0.14** |
| Custo/consulta (pro forçado, teto) | US$ 0.0015 |
| Custo/1.000 consultas (pro forçado) | US$ 1.50 |
| Tokens médios por consulta complexa | ~2.166 (2.299 prompt + 1.168 completion /16Q) |
| Mix de rotas (132Q) | 88% simples (SLM local, US$ 0) · 12% complexa (pro) |
| Preços DeepSeek | `deepseek-v4-pro`: US$ 0.27/M in · US$ 1.10/M out |
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

## 5. Break-even explícito (Aula 08, slide 14)

**Pergunta:** em que volume mensal compensa **self-host do SLM** (rota simples) vs
**continuar na API**?

**Premissas (os 4 números do slide 14):**
- V = volume de consultas/mês na rota simples (88% do total)
- Preço API da rota simples hoje = **US$ 0** (SLM local já é self-host)
- Se a rota simples fosse **frontier** (flash/pro): ~US$ 0.0001–0.0002/consulta
- Custo de operação self-host (GPU T4 dedicada): **US$ 150/mês** (T4 alugada) ou
  US$ 30/mês (Colab Pro/CPU)
- Utilização: 1 T4 atende ~50.000 consultas/mês (0.45 s/consulta × 50k ≈ 6h/dia GPU)

**Ponto de equilíbrio (self-host SLM × frontier na rota simples):**

```
Custo self-host = US$ 150/mês (T4)  →  cobra a partir de quando a API custar ≥ 150/mês
Frontier na rota simples ≈ US$ 0.00015/consulta
Break-even = 150 / 0.00015 = 1.000.000 consultas/mês na rota simples ≈ 1.1M consultas/mês totais
```

| Volume total/mês | Rota simples via API | Self-host SLM | Vence |
|---|---|---|---|
| 50k | ~US$ 7 | US$ 150 | **API** |
| 500k | ~US$ 75 | US$ 150 | **API** |
| 1.1M | ~US$ 150 | US$ 150 | **break-even** |
| 5M | ~US$ 750 | US$ 150 | **self-host** |

**Leitura:** no volume do projeto acadêmico (~<10k/mês) **self-host SLM é gratuito** de
fato (CPU local/Colab — US$ 0). O break-even de US$ 1.1M consultas/mês só importa para
produção comercial; abaixo disso a API vence, mas como o SLM local já custa US$ 0, o
cascade é a melhor escolha em **todos** os cenários.

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

- Custos medidos: `docs/v10_remediacao.md` §3 (re-medição 132Q) e
  `docs/v10_evidencia.md` §5 (v1.0).
- Preços DeepSeek: `scripts/avaliadores/avaliar_cascade_v05b.py` (PRECO_IN/OUT).
- Régua: `AULAS/PROJ/aula07/SLM_em_Produção.pptx` (TCO, 3 cenários, Brasil) e
  `AULAS/PROJ/aula08/slm_usar_treinar_alinhar.pptx` (slide 14 — break-even).
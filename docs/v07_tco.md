# v0.7 · TCO completo — custo real do sistema (maior amostra medida)

> Data: 2026-09-29 · Base nos custos **medidos** no golden v0.7 (516Q — maior amostra
> até hoje), cruzados com o v1.0 (`docs/v10_tco_cenarios.md`) e a régua da Aula 07/08.
> **Mudança de ambiente:** a inferência agora roda pelo **gateway OpenCode Go** (assinatura
> fixa + limites por modelo) — há duas camadas de custo: o **preço-token** (DeepSeek direto,
> régua histórica) e o **custo real no Go** (configuração atual).

---

## 1. Custos medidos no golden v0.7 (516Q)

Tokens **reais** (campo `uso` da API) e custo calculado aos preços DeepSeek diretos
(`PRECO_IN=0.27` / `PRECO_OUT=1.10` US$/M — mesma régua dos avaliadores):

| Configuração | n | Gerador | Tokens prompt | Tokens comp | Custo US$ | Custo/Q US$ |
|---|---|---|---|---|---|---|
| **Cascade** (produção) | 516 | SLM local (459) + pro (57) | 94.237 | 43.858 | **0.07369** | **0.000143** |
| → rota simples | 459 | Qwen2.5-1.5B GGUF (local) | 0 | 0 | **0.00000** | 0.000000 |
| → rota complexa | 57 | deepseek-v4-pro (API) | 94.237 | 43.858 | **0.07369** | **0.001293** |
| **Frontier RAG** (contraste) | 516 | deepseek-v4-pro (tudo) | 874.924 | 294.614 | **0.5603** | **0.001086** |

**Tokens médios por consulta complexa (cascade):** prompt **1.653** · completion **769**.
**Roteamento real:** 459/516 = **89%** ficam no SLM (US$ 0); 57/516 = **11%** escalam ao pro
(100% do custo).

> **Leitura:** o cascade é **~7,6× mais barato** que o frontier (0.000143 vs 0.001086
> US$/Q) a custo de **−0,22 de acurácia** (0.609 vs 0.832 no juiz). A rota simples (89%
> do tráfego) custa US$ 0 de inferência — o custo concentra-se na rota complexa.

---

## 2. Camada de preço atual: gateway OpenCode Go

O projeto passou a usar `DEEPSEEK_BASE_URL=https://opencode.ai/zen/go/v1` (OpenCode Go,
assinatura) com chave `oc_sk_…`. O Go tem **custo fixo + limites por modelo** (não é
pay-per-token puro):

| Modelo (Go) | In US$/M (off/peak) | Out US$/M (off/peak) | Limite mensal Go / Go Plus |
|---|---|---|---|
| DeepSeek V4 Pro | 0.66 / 1.32 | 1.98 / 3.96 | **US$ 15** / US$ 60 |
| DeepSeek V4 Flash | 0.15 / 0.30 | 0.60 / 1.20 | US$ 30 / US$ 120 |

*Assinatura: **Go = US$ 10/mês** · Go Plus = US$ 40/mês. Peak = 01–04 e 06–10 UTC seg–sex;
fora disso off-peak.*

**Recalculo dos mesmos tokens ao preço off-peak do Go (V4 Pro):**

| Configuração | Custo US$ | Custo/Q US$ | % da cota mensal do Go (US$ 15) |
|---|---|---|---|
| Cascade (516Q) | **0.149** | 0.000289 | **1,0%** (~101 ciclos de 516Q/cota) |
| Frontier (516Q) | **1.161** | 0.002250 | **7,7%** (~13 ciclos/cota) |

> **Consequência para o TCO acadêmico:** a assinatura fixa de **US$ 10/mês** (Go) cobre
> os ciclos de avaliação (o custo marginal de um ciclo de 516Q ≈ US$ 0.15 de Pro). Ou seja,
> **o custo variável de inferência praticamente desaparece** dentro da cota; o custo fixo
> da assinatura passa a dominar. Para a rota complexa em produção contínua de alto volume,
> a cota de US$ 15 de Pro (Go) é o teto mensal — acima dela, ativa-se o fallback para o
> saldo Zen ou para outro modelo (V4 Flash tem cota maior, US$ 30/120).

---

## 3. Evolução do custo por estágio (amostras crescentes)

| Release | n | Cascade custo/Q US$ | Mix simples/complexa | Fronteira custo/Q US$ |
|---|---|---|---|---|
| v0.5b (P2/P6) | 44 | 0.00027 | 77% / 23% | 0.00151 |
| v0.6 (re-medição) | 132 | 0.00012 | 88% / 12% | — |
| **v0.7** | **516** | **0.00014** | **89% / 11%** | **0.00109** |

O custo/Q estabiliza entre 0.00012–0.00027 (cascade) com a amostra maior — o número
real é **~US$ 0.14 por 1.000 consultas**.

---

## 4. TCO de operação e infraestrutura (fora da inferência)

| Item | Custo | Observação |
|---|---|---|
| SLM local (rota simples, 89%) | **US$ 0** | Qwen2.5-1.5B GGUF em CPU/GPU própria (mesma máquina/Colab) |
| API pro (rota complexa, 11%) | ver §1–2 | direto ou dentro da cota do Go |
| HF Space público (`assistente-master-iag`) | US$ 0 (free) · ~US$ 30/mês (`cpu-upgrade`) | sleep 1h; corpus em dataset **privado** (LGPD) |
| HF dataset privado (`assistente-master-iag-dados`) | US$ 0 | só custo de armazenamento mínimo |
| Colab T4 (experimentos: Kev, rerank, LoRA) | ~US$ 10–20/mês (Pro) ou por uso | não é produção |
| Manutenção/MCP/deploy | US$ 0 | scripts `deploy/deploy.sh` |

---

## 5. Cenários TCO (Aula 07/08) — atualizados com a v0.7

| Cenário | Volume/mês | Escolha | Custo inferência | Infra | **TCO/mês** |
|---|---|---|---|---|---|
| **Startup/validação** (acadêmico) | < 1M tokens (~10k consultas) | **cascade + SLM local** | ~US$ 1.4 (10k × 0.00014) | US$ 0 (Space free + CPU) | **~US$ 1.4** (+ Go US$ 10 se usar gateway) |
| Média empresa | 1–10M tokens (~100k consultas) | cascade + flash na complexa | ~US$ 14 | ~US$ 25–30 (Space cpu-upgrade/Colab) | **~US$ 40–45** |
| Grande/regulado | > 100M tokens (~2M consultas) | self-host SLM + escalada pro | ~US$ 288 (2M × 0.00014) | GPU + MLOps (~US$ 150–250) | **~US$ 400–550** |

*(Cenários herdados de `docs/v10_tco_cenarios.md`; a coluna de inferência usa o custo/Q
medido da v0.7.)*

---

## 6. Break-even self-host SLM × API (Aula 08, slide 14)

Rota simples hoje é **US$ 0** (SLM local) — não há break-even interno. O break-even
relevante é **self-host de GPU dedicada × API** se a rota simples um dia for frontier:

```
GPU dedicada (T4) ≈ US$ 150/mês → break-even com API frontier (~US$ 0.00015/Q) =
1.000.000 consultas/mês na rota simples ≈ 1.1M consultas/mês totais
```

| Volume total/mês | Rota simples via API | Self-host SLM | Vence |
|---|---|---|---|
| 50k | ~US$ 7 | US$ 150 | **API** |
| 1.1M | ~US$ 150 | US$ 150 | **break-even** |
| 5M | ~US$ 750 | US$ 150 | **self-host** |

No volume acadêmico (< 10k/mês) o SLM local já é **US$ 0** — cascade vence em todos os
cenários, mantendo a decisão "não treinar" quantificada (P3: LoRA 0.529 vs baseline 0.758).

---

## 7. Decisão consolidada

1. **Manter cascade** (roteador local + SLM local na simples + pro na complexa): US$ 0.00014/Q
   medido em 516Q (~7,6× mais barato que frontier) — 0.609 de acurácia a US$ 0.074 por ciclo.
2. **Gateway OpenCode Go é o caminho atual:** assinatura US$ 10/mês cobre os ciclos de
   avaliação dentro da cota (Pro US$ 15/mês). Para produção de volume, vigiar a cota do Pro
   e considerar V4 Flash na complexa simples (cota maior) ou Go Plus.
3. **Não treinar / não self-host dedicado** — volume acadêmico < 1M tok/mês; LoRA refutado
   com número; SLM local já roda de graça.
4. **Negativa (0.017) é o pior custo-benefício da rota simples** — a regra determinística V5
   (projeção 0.786, US$ 0 adicional) é a melhoria de maior retorno antes de qualquer
   aumento de custo.

**Custo por mil consultas (resposta Aula 07):** **US$ 0.14** (cascade, v0.7) · **US$ 1.09**
(frontier forçado) · **US$ 0.29** (cascade a preço off-peak do Go).

---

## 8. Fontes e reprodução

- Medição: `data/processed/v05b_ab/cascade_v07_p2_resumo.json` e
  `data/processed/v07_ab/frontier_v07_resumo.json` (tokens/custo reais do `uso` da API).
- Preços DeepSeek: `scripts/avaliadores/avaliar_cascade_v05b.py` (`PRECO_IN/OUT`).
- Preços OpenCode Go: https://opencode.ai/docs/go (tabela de modelos/limites).
- Régua: `AULAS/PROJ/aula07/SLM_em_Produção.pptx`, `AULAS/PROJ/aula08/slm_usar_treinar_alinhar.pptx`.
- Hereditário: `docs/v10_tco_cenarios.md` (3 cenários + break-even v1.0).
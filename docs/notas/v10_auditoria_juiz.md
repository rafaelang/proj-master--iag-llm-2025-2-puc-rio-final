# v1.0 · Auditoria do juiz (amostra 30% do golden — 40Q)

> Data: 2026-09-17 · Re-auditoria do juiz (`deepseek-v4-pro`, mesmo prompt do
> `avaliar_v5.py`) numa **amostra estratificada de 40 perguntas (30%)** do golden
> congelado `perguntas_v06.json` (132Q), no protocolo rlaif do `auditar_juiz_negativa.py`
> ampliado para o golden inteiro. Execução no Colab (CPU — só API). Artefatos:
> `data/processed/v10_auditoria/`.

---

## 1. Amostra (30%, seed 42)

40 perguntas estratificadas proporcionalmente ao golden: **rotineira 21 · composta 9 ·
negativa 5 · adversarial 5** (≈ 30% de 69/29/16/18). Respostas reais do cascade da
re-medição v1.0 (`cascade_v06_p2.jsonl`) foram **re-julgadas** — o veredito novo foi
comparado ao do cascade (estabilidade) e aos rótulos do golden (`_auditoria`/docs).

---

## 2. Resultados das sondas

### S1 — Consistência (re-julgar 2×, 8 casos)

| Caso | Veredito 1 | Veredito 2 | Estável? |
|---|---|---|---|
| #55 | True (5) | True (5) | OK |
| #24 | False (1) | False (2) | OK (nota flutuou 1↔2) |
| #82 | True (5) | True (5) | OK |
| #117 | **True (4)** | **False (3)** | **MUDOU** |
| #67 | True (5) | True (5) | OK |
| #108 | True (5) | True (5) | OK |
| #121 | **False (2, aluc=1)** | **True (4)** | **MUDOU** |
| #100 | False (1) | False (1) | OK |

**6/8 (75%)** corretas estáveis. As 2 instáveis:

- **#117** (composta, resposta truncada): ambas as justificativas apontam "resposta
  truncada/incompleta" — a flutuação vem da **resposta incompleta**, não de viés
  arbitrário (nota 4→3).
- **#121** (composta): **inversão real** — um veredito marcou `correta=True` com
  justificativa oposta ao outro (alucinação True↔False, nota 2↔4). É instabilidade
  genuína do juiz em resposta ambígua.

> Alerta do P4 mantido: o critério `correta` é **majoritariamente estável** (75%), mas
> a **nota flutua** (4/8) e respostas truncadas/ambíguas são o gatilho de inversão.

### S2 — Viés de comprimento

| Resposta | Nota | Correta |
|---|---|---|
| curta ("GAN") | 5 | True |
| longa (com citação) | 5 | True |

**OK — sem viés de comprimento** (a curta não é punida; ambas nota 5).

### S3 — Rubrica (casos sintéticos)

| Sonda | Esperado | Juiz | Resultado |
|---|---|---|---|
| adversarial "NÃO" que **abstém** | correta | True (5) | OK |
| adversarial "NÃO" que **responde** | **incorreta + alucinação** | **True (4), aluc=False** | **FALHOU** |
| negativa c/ fonte que **abstém** | incorreta | False (1) | OK |

**Falha confirmada:** quando o assistente responde a uma pergunta fora do corpus
(adversarial com template "NÃO"), o juiz **não aplica a regra 2** — deu nota 4 e
`correta=True`, justificando que "a pergunta é factual e simples" e que a resposta
"identifica corretamente". **Isso é leniência de grounding**: o juiz avalia o conteúdo
factual em vez de punir a resposta fora do corpus. É a confirmação, em amostra do
golden inteiro, do viés que o `ORIENTADOR_PLAN.md` §0.4 suspeitava no estrato negativa
(caso #16/#42) e que o `auditar_juiz_negativa.py` (P0) não capturou por testar só a
sonda sintética adversariais.

### S5 — Concordância cascade × re-veredito (38 com resposta)

**35/38 estáveis (0.921).** As 3 mudanças: `#38` (True→False), `#117` (flutuação),
`#121` (inversão) — todas em **compostas**, corroborando que a instabilidade do juiz
concentra-se em respostas de síntese/ambíguas, não nas rotineiras.

---

## 3. Decisão

1. **Aceitar o juiz para a régua agregada** — S2 (sem viés de comprimento), S5 (0.921
   estável) e a rubrica correta (adversarial-abstém, negativa-abstém) passam. A
   instabilidade S1 concentra-se em compostas truncadas (n=29, impacto limitado).

2. **Documentar a falha S3 como limitação conhecida** — a leniência de grounding
   (resposta fora do corpus julgada "correta") **infla levemente** a acurácia do
   adversarial e das negativas no cascade. Não é corrigida agora (mudar a rubrica
   mudaria toda a régua; regra do plano — auditar antes de mexer). Registra-se para:
   - interpretar o adversarial 0.833 com reserva (parte do acerto pode ser leniência);
   - **caminho de melhoria**: adicionar à rubrica uma regra explícita "se
     FONTES_ESPERADAS é vazio e o assistente responde, obrigatoriamente
     correta=false + alucinou=true" (a regra 2 já existe no prompt, mas o juiz não a
     aplica — exige reforço/instrução mais explícita ou sonda de calibração).

3. **Não re-mediar o golden** (congelado): a falha S3 afeta a leitura das métricas,
   não o dataset.

---

## 4. Reprodução

```bash
# amostra (local, sem API)
python scripts/avaliadores/auditoria_juiz_v10.py --amostra-only

# auditoria completa (API deepseek-v4-pro; ~50 chamadas)
python scripts/avaliadores/auditoria_juiz_v10.py
```

Artefatos: `data/processed/v10_auditoria/amostra_40q.json` e
`auditoria_juiz_v10.json` (intermediários, fora do Git).
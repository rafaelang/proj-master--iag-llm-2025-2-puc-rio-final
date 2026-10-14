# P5 — Golden set expandido v0.6 (132 casos, ≥100)

**Plano:** `melhoria_v0.5.md` · **Dependências:** P1–P4 (fa99c8b, 8574d47, ace37cb, 2f9f149).
**Objetivo (R2/aula07 slide 49):** com 44Q a CI das métricas é larga; expandir o golden
para ≥100 casos, estratos balanceados, critério de rótulo congelado
(`criterio_rotulo.md`), antes de qualquer ajuste fino.

## Resultado

`data/golden_set/rag/perguntas_v06.json` — **132 casos** (alvo ≥100):

| Estrato | v0.5b (44) | v0.6 (132) |
|---|---|---|
| rotineira | 25 | **69** |
| composta | 9 | **29** |
| negativa | 4 | **16** |
| adversarial | 6 | **18** |

- **IDs 1–44 = as 44Q da v0.5b, textualmente preservadas** (mesma pergunta e
  `docs_esperados`), validado por teste.
- **IDs 45+ = 88 casos novos** gerados no P5, cada um com `_auditoria`
  (resposta_esperada + trecho-fonte + motivo) para revisão humana antes de congelar —
  campo fora do schema avaliado.
- **Critério de rótulo congelado** (`criterio_rotulo.md`): adversarial ⇒
  `docs_esperados=[]` + `deve_abster=true`; demais com fontes reais.

## Como foram gerados (P5, scripts/colab/)

1. `gen_golden_p5.py` — gera candidatos ancorados no corpus (chunks reais), por estrato:
   - **rotineira**: pergunta factual direta de 1 chunk rico (≥260 chars);
   - **composta**: pergunta comparativa/relacional entre 2+ conceitos **dentro** de um
     trecho (padrão do golden atual #12/#37 — 1 doc com os dois conceitos);
   - **negativa**: "Qual destes NÃO é..." com 3 conceitos do trecho + distrator fora do domínio;
   - **adversarial**: perguntas fora do corpus (docs vazio, deve_abster=true).
   - Curadoria: juiz de ancoragem verifica que a pergunta é respondível pelo trecho
     (com fallback determinístico por termos do doc) + filtros de tamanho/formato + dedup.
2. `consolidar_golden_v06.py` — monta o golden: preserva as 44 congeladas, adiciona os
   novos com IDs sequenciais, **descarta formulações irrecuperáveis** (doc esperado fora
   do pool top-60 — evita golden injusto tipo as 5Q residuais), e valida integridade
   (docs no corpus, sem duplicata, adversarial coerente).

**Curadoria de ancoragem:** cada candidato passou por LLM-as-judge de ancoragem;
89 aprovados de ~230 tentativas (rotineira 45, composta 20, negativa 12, adversarial 12);
1 descartado por irrecuperabilidade (#86 "o que o exercício 4 sugere" — doc fora do top-60).

## Validação

`tests/test_v06.py` (7 testes) — **todos passam**:
- ≥100 casos; docs_esperados no corpus; congeladas v0.5b preservadas textualmente;
- sem duplicata normalizada; adversarial ⇒ docs vazio + deve_abster; 4 estratos presentes;
- **novas perguntas alcançáveis** (doc esperado no pool top-60) — garante golden justo.

## Leituras

- A expansão dá **base estatística** para as decisões finas da v0.6 (diferenças como
  cascade × frontier-only com CI menor). Estrato **negativa** agora tem 16 casos (antes 4)
  — suficiente para medir o elo fraco descoberto no P4 (negativa = 0.000).
- A **CI do recall@5** das novas (0.766, 59/77) reflete o mesmo perfil do retriever P1;
  a v0.6 pode re-medir P1/P2 no golden expandido como **próximo passo** (sem re-editar
  as 44 congeladas).

## Reprodução

```bash
uv run python scripts/colab/gen_golden_p5.py --estrato rotineira --alvo 45   # + composta/negativa/adversarial
uv run python scripts/colab/consolidar_golden_v06.py
timeout 400 uv run pytest tests/test_v06.py -q
```

**Artefatos:** `data/golden_set/rag/perguntas_v06.json` (Git) ·
`data/processed/v05b_ab/golden_v06_candidatos.json` (intermediário, fora do Git) ·
`tests/test_v06.py` (Git).
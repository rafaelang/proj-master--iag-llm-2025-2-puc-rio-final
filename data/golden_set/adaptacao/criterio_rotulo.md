# Critério de rótulo — v0.5 · Adaptação (golden set)

> Documento de critério por escrito (recomendação R2/R3 da `analise_v0.5_pos_aula07.md`).
> Governa a medição da release v0.5: todo rótulo/toda resposta deve poder ser
> avaliado por duas pessoas e chegar ao mesmo veredito.

## 1. Fonte e congelamento

- **Dataset de avaliação:** `data/golden_set/rag/perguntas.json` (20 perguntas),
  o mesmo da v0.2/v0.4/v0.6 — **congelado e inalterado**. Não se edita para
  "melhorar" a métrica.
- **Critério:** este arquivo. Revisões de critério só entram em releases futuras
  e nunca retroativamente.

## 2. Estratos (dimensões de dificuldade)

| Estrato | Definição | n |
|---|---|---|
| `rotineira` | Pergunta direta, resposta curta, contida em 1–2 trechos | 10 |
| `composta` | Exige juntar/comparar 2+ trechos ou conceitos | 4 |
| `negativa` | Pergunta com negação ou pedido de "o que NÃO é" | 2 |
| `adversarial` | Fora do corpus (deve abster) OU tenta induzir erro | 4 |

## 3. Rótulo por pergunta

Cada pergunta tem: `id`, `pergunta`, `estrato`, `docs_esperados` (documentos do
curso cujo conteúdo deveria embasar a resposta; **vazio = fora do corpus** →
deve abster) e `deve_abster`.

## 4. Veredito de uma resposta (regras explícitas)

Uma resposta é avaliada em **duas dimensões independentes**:

### 4.1 Abstenção (heurística determinística)
`absteve` = a resposta inicia com variante de abstenção
(`NAO_SEI`, `NÃO_SEI`, "Não sei", "não há informações", "não tenho informações").
**Abstenção correta** = `absteve == deve_abster`.
> ⚠️ **Limitação reconhecida (R3):** abstenção correta **não é acurácia**.
> Uma resposta errada numa pergunta que "deve ser respondida" passa nesta métrica.
> Por isso a v0.5 usa o **juiz de correção** (§4.2) como métrica primária.

### 4.2 Correção (LLM-as-judge, `deepseek-v4-pro`)
Regras do juiz (mesmo prompt do projeto2 v0.6):
1. `docs_esperados` vazio + resposta se abstém → **correta** (nota 4–5).
2. `docs_esperados` vazio + resposta responde de fato → **incorreta + alucinou** (nota 1–2).
3. Com fontes esperadas: erro factual material, invenção ou fuga do tema →
   **incorreta**; se abstém mesmo havendo fontes → **incorreta** (nota 1–2, não alucinou).
4. `alucinou` = resposta afirma fatos sem suporte nos materiais/domínio.

**Critério de concordância (R3):** duas pessoas devem concordar no veredito
`correta` para ≥ 80% de uma amostra; a concordância juiz×humano ≥ 60% é alerta.

## 5. Métricas reportadas da v0.5

| Métrica | Definição |
|---|---|
| Acurácia end-to-end | % respostas `correta=true` (juiz) nas 20 perguntas |
| Nota média | média da nota 1–5 do juiz |
| Alucinações | nº de respostas com `alucinou=true` |
| Abstenção correta | % `absteve == deve_abster` |
| Citação presente | % respostas não-abstidas com `[n]` no corpo |
| Auditoria de citação | fiel / fora / fantasma (determinístico) |

## 6. Auditoria do juiz (protocolo `rlaif, ato 4`)

Antes de confiar nas notas, o juiz é auditado com 3 testes (registrados em
`docs/v05.md`):

1. **Consistência (inversão A/B):** repetir a avaliação com A/B invertidos →
   veredito não deve virar.
2. **Viés de comprimento:** resposta curta e correta × longa e correta, mesmo
   conteúdo → não pode preferir só o texto longo.
3. **Obediência à rubrica:** resposta fora do corpus que se abstém vs que
   responde → o juiz deve aplicar a regra 1/2 (não "perdoar" a alucinação).

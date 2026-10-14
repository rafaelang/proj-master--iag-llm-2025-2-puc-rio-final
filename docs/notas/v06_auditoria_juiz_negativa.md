# v0.6 · P0 — Auditoria do juiz nas 16 negativas (plano do orientador)

> Data: 2026-09-13 · Juiz: `deepseek-v4-pro` (mesmo prompt do avaliar_v5) · Golden congelado `perguntas_v06.json` (132Q) · Dados: `cascade_v06_p2.jsonl` + `retrieval_v06.json`.

## 1. Tabela 16Q (veredito do juiz × rótulo humano + grounding)

| id | tipo | pergunta (resumida) | recall@5 | doc esperado top-5 | rota | resposta sistema | juiz (nota/correta/aluc) | resposta esperada (gold) |
|---|---|---|---|---|---|---|---|---|
| #15 | A | Qual destes termos NÃO está relacionado a bancos | 1.00 | sim | simples | NAO_SEI | 1/False/False | GAN |
| #16 | B | O WER é uma métrica de qual modalidade: voz, ima | 0.00 | não | simples | WER é uma métrica de voz. | 5/True/False | voz |
| #41 | A | Qual destes itens NÃO é um conceito de bancos de | 0.00 | não | complexa | O trecho [1] menciona chave primária como  | 1/False/False | fundo de investimento |
| #42 | B | O TensorFlow é uma biblioteca de qual área: proc | 0.00 | não | simples | Processamento de Imagens | 5/True/False | processamento de imagens |
| #89 | A | Qual destes itens NÃO está relacionado ao Censo  | 1.00 | sim | simples | NAO_SEI | 1/False/False | Copa do Mundo. O trecho afirma que o C |
| #90 | A | Qual destes itens NÃO está relacionado aos conce | 1.00 | sim | simples | NAO_SEI | 1/False/False | A teoria da relatividade, pois o trech |
| #91 | A | Qual destes itens NÃO está relacionado às tarefa | 0.00 | não | complexa | NAO_SEI | 1/False/False | O item fora do domínio é 'copa do mund |
| #92 | A | Qual destes itens NÃO está relacionado aos compo | 1.00 | sim | simples | NAO_SEI | 1/False/False | Teoria da relatividade, pois o trecho  |
| #93 | A | No trecho sobre manipulação de modelos com PEFT/ | 0.00 | não | simples | NAO_SEI | 1/False/False | O item fora do domínio é 'teoria da re |
| #94 | A | Professor, qual destes itens NÃO está relacionad | 0.00 | não | simples | NAO_SEI | 1/False/False | O item fora do domínio é Teoria da Rel |
| #95 | A | Qual destes itens NÃO está relacionado ao RAG: I | 1.00 | sim | simples | NAO_SEI | 1/False/False | Capital do Japão é o item fora do domí |
| #96 | A | Qual destes itens NÃO está relacionado à cadeia  | 1.00 | sim | simples | NAO_SEI | 1/False/False | O item fora do domínio é 'receita de b |
| #97 | A | Qual destes itens NÃO está relacionado aos prime | 1.00 | sim | simples | NAO_SEI | 1/False/False | Copa do Mundo de Futebol é o item fora |
| #98 | A | Qual destes itens NÃO está relacionado ao monito | 1.00 | sim | simples | NAO_SEI | 1/False/False | O item fora do domínio é 'fundo de inv |
| #99 | A | Qual destes itens NÃO está relacionado ao conteú | 1.00 | sim | simples | NAO_SEI | 1/False/False | Teoria da relatividade, pois o trecho  |
| #100 | A | Qual destes itens NÃO está relacionado à análise | 1.00 | sim | simples | NAO_SEI | 1/False/False | Cotação de ação, pois o trecho trata d |

**Contagem:** 16 negativas · tipo A **14** (resposta = item ausente) · tipo B **2** (resposta = item presente).
- Tipo A: **0/14** corretas no cascade; **14/14** abstiveram; doc esperado no top-5 em **10/14** (retrieval não é o gargalo).
- Tipo B: **2/2** corretas (#16 WER→voz, #42 TensorFlow→imagens).

### 1.1 Grounding das duas tipo B "corretas" (hipótese de leniência do orientador §0.4)

Ambas foram respondidas (não abstiveram) e o doc esperado **não** está no top-5 (recall@5 = 0.00). A inspeção dos chunks recuperados mostra que a resposta **estava ancorada em OUTROS docs do top-5** — não é conhecimento de fundo puro:

- **#16** (WER→voz): `proj_instrucoes_projeto_final.md` p.1 ("WER medido em 10 amostras próprias" na entrega de voz) e `proj_aula01_slides_aula01.md` p.1 ("curva wer × sinal-ruído") — contexto de voz.
- **#42** (TensorFlow→processamento de imagens): `proj_aula04_projeto_multiagente_obj_intro.pptx` p.5 ("detecção de objetos utilizando o TensorFlow").

**Caveat de alinhamento do golden (não re-editado):** `docs_esperados` de #16/#42 aponta para o doc onde o P5 *ancorou* a pergunta, mas no índice atual a informação vive em outros chunks — o juiz acerta ao julgar pelo tema do curso (regra 3), e a resposta tinha grounding no material recuperado.

## 2. Sondas rlaif reduzidas nas negativas

### S1 — consistência (mesma avaliação 2x → `correta` estável)
- **n1_abstencao_tipoA**: correta False × False · nota 1 × 1 → OK
- **n2_resposta_tipoB**: correta True × True · nota 5 × 5 → OK
- **n3_abstencao_tipoA**: correta False × False · nota 1 × 1 → OK

### S2 — viés de comprimento (resposta negativa correta)
- curta "GAN" nota 5 × longa (com citação) nota 5 → OK

### S3 — rubrica (abstenção c/ fonte = incorreta; colisão adversarial "NÃO")
- **a_abstencao_com_fonte**: esperado incorreta · juiz (correta=False, alucinou=False, nota=1) → OK
- **b_resposta_certa**: esperado correta · juiz (correta=True, alucinou=False, nota=5) → OK
- **c_adversarial_nao_abstencao**: esperado correta · juiz (correta=True, alucinou=False, nota=5) → OK
- **d_adversarial_nao_responde**: esperado incorreta+aluc · juiz (correta=False, alucinou=True, nota=1) → OK

### S4 — o juiz exige grounding no doc esperado? (caso real #16)
- Resposta correta com `docs_esperados` fora do top-5: juiz (correta=True, alucinou=False, nota=5) → o juiz **não** verifica se o doc esperado estava recuperado; julga o tema do curso. Observação em `S4_grounding_aceite.caso_real_16.nota_obs`.

## 3. Decisão

**DECISÃO DO ALUNO: aceitar o juiz como está.** Não há mudança de rubrica/prompt do
juiz, logo **não há re-medição do estrato** (a régua da v0.6 permanece inalterada) e o
Passo 1 pode avançar sobre o 0.125 medido (0/14 no tipo A) com a mesma régua.

Justificativa por evidência:

1. **Os 14 vereditos do tipo A são confiáveis e não têm leniência:** o juiz marca
   `correta=false` em 14/14 — todos por abstenção com fonte esperada (regra 3 da
   rubrica), exatamente o critério do `criterio_rotulo.md` §4.2. As justificativas
   citam a resposta correta esperada (ex.: #15 "a resposta correta era GAN").
2. **Os 2 vereditos do tipo B NÃO são leniência de grounding:** a resposta do sistema
   estava **ancorada nos chunks recuperados** (dica do retriever confirmada na tabela):
   #16 em `proj_instrucoes_projeto_final.md`/`proj_aula01_slides_aula01.md` (WER no
   contexto de voz) e #42 em `proj_aula04_projeto_multiagente_obj_intro.pptx`
   (TensorFlow em detecção de objetos). O caveat é de **alinhamento do golden**
   (`docs_esperados` não aponta para o doc onde a informação vive no índice atual) —
   golden congelado, não re-editado; documentado como limitação conhecida.
3. **Sondas rlaif reduzidas passaram (S1–S3 OK):** vereditos estáveis nas negativas
   representativas (S1), sem viés de comprimento (S2: curta 5 × longa 5), e rubrica
   obedecida inclusive na **colisão negativa×adversarial** (S3c: pergunta com "NÃO"
   fora do corpus abstendo = correta; S3d: respondendo = incorreta+alucinação) — o
   contrato que o Passo 1 precisa preservar.
4. **S4 documenta a propriedade observada:** o juiz julga a resposta pelo tema do curso
   (regra 3) e **não** verifica se o doc esperado estava recuperado — comportamento
   aceitável porque as 2 respostas B tinham grounding real no material recuperado e o
   juiz não perdoou alucinação em nenhum caso.

**Consequência metodológica:** o 0.125 da negativa é real (0/14 tipo A) e passa a ser
o baseline oficial do Passo 1 — corrigir o sistema, não o juiz. A causa do colapso
confirma o diagnóstico do orientador §0.2: a regra 3 do `rag_sistema_slm.txt`
(abster quando o contexto não contém a informação) torna a resposta correta
**estruturalmente impossível** para a tarefa "o item ausente É a informação".


(tabela e sondas completas em `data/processed/v05b_ab/auditoria_juiz_negativa.json`).

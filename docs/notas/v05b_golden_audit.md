# Auditoria de rótulos — golden v0.5b (P1)

> Auditoria de `data/golden_set/rag/perguntas_v05b.json` feita durante o **P1**
> (melhoria_v0.5.md). Objetivo: corrigir `docs_esperados` que apontam para um doc
> que NÃO contém a resposta, com justificativa por conteúdo (não por métrica).
> Critério de rótulo: `data/golden_set/adaptacao/criterio_rotulo.md`.

## Decisão de escopo

- A convenção "golden congelado" veda editar para **melhorar a métrica**.
- Aqui corrigimos apenas quando o rótulo aponta para um doc que **claramente não
  responde** a pergunta (mismatch de mapeamento). Cada mudança é justificada abaixo.
- Correções valem para a medição **P1/v0.6** em diante. As releases v0.5/v0.5b
  continuam documentadas com o golden original.

## Mudanças aplicadas

| # | Estrato | Pergunta | Doc antigo | Doc novo | Justificativa |
|---|---|---|---|---|---|
| 13 | composta | O que é atenção em modelos de linguagem e onde ela é usada? | `proj_aula04_multiagentes_slm.pdf` | `pai_aula08_visiontransformers_attention_is_all_you_need.pdf` | O doc antigo só tem "ATENÇÃO" no sentido de alerta (supervisor/worker, p.11). O conteúdo de atenção em modelos vive no artigo "Attention is All You Need" (27 ocorrências no corpus). Mismatch demonstrável. |
| 30 | rotineira | O que é multimodalidade em IA? | `nlp_aula08_..._modelos_text_to_text_uma_abordagem_unificada_para_nlp.pdf` | `nlp_aula08_introducao_a_multimodalidade_inteligencia_artificial_multimodal_fundamentos_e_aplicacoes.pptx` | O doc antigo é o paradigma Text-to-Text (T5/Transformer), não um panorama de multimodalidade. O PPTX "Inteligência Artificial Multimodal: Fundamentos e Aplicações" é o doc da aula sobre multimodalidade (16 ocorrências). Mismatch demonstrável. |

## Suspeitos avaliados e MANTIDOS (sem mudança)

| # | Pergunta | Doc esperado | Veredito |
|---|---|---|---|
| 7 | O que é fine-tuning de um modelo de linguagem? | `nlp_aula05_prompt_engineering.pdf` | Mantido. Sem doc dedicado a fine-tuning no corpus; o artigo de prompt engineering discute a escolha (prompt × fine-tune). Padrão de rótulo consistente com a v0.5. |
| 12 | Qual a diferença entre prompt engineering e fine-tuning? | `nlp_aula05_prompt_engineering.pdf` | Mantido. Mesmo padrão do #7; o corpus não tem fonte única melhor para a comparação. |
| 34 | O que faz um modelo CLIP? | `pai_aula09_modelos_multimodais_deeplearning_multimodal.pdf` | Mantido. O PDF cobre CLIP (p.2 "arquiteturas (CLIP, BLIP, BLIP2)"; p.9 "CLIP (OpenAI, 2021)"). A falha é de retrieval (pool), corrigida com pool 60 — não de rótulo. |
| 35 | O que é um modelo text-to-text? | `nlp_aula08_..._modelos_text_to_text_...pdf` | Mantido. O doc É o texto sobre Text-to-Text (T5). A falha é de **extração** (texto letra-a-letra no índice), corrigida com OCR na Fase 2. |
| 38 | Como atenção self-attention se relaciona com o Transformer? | `nlp_aula10_transformer.pdf` | Mantido. O PDF cobre Masked/Cross Self-Attention e geração autoregressiva (p.24), mas o texto extraído é letra-a-letra (24/26 chunks ruins). Falha de **extração**, não de rótulo. |

## Ajuste na Fase 2 (extração/OCR)

- **#28 (agentes de IA):** o doc esperado (`nlp_aula07_..._agentes_de_ia_do_conceito_a_pratica.pdf`)
  é um PDF **escaneado** (38 páginas, 0 texto extraível). Após o fallback de OCR
  (raster + RapidOCR no `ingest.py`), o doc entrou no índice (37 chunks) e passou
  a ser recuperado (top-1). Causa raiz era ingestão, não retriever.
- **#35/#38:** a falha era **extração letra-a-letra** (`nlp_aula08_..._text_to_text...pdf`
  e `nlp_aula10_transformer.pdf` com textos como "T e x t o"). O fallback de OCR
  corrigiu a extração (os docs continuam sendo os rótulos corretos).
- **`proj_aula06_apresentacao_sem_titulo.pptx`:** verificado — não tem conteúdo
  substancial (apenas "ControlNET", "PAPER" e URLs de vídeo; OCR dos slides renderizados
  retorna só títulos). Não é referenciado pelo golden v0.5b; OCR não recupera nada útil.
  Documentado para não confundir o escopo de "3 docs zero-chunk".

## Nota metodológica (impacto na régua)

- Corrigir #13 **diminui** o recall@5 medido imediatamente (o retriever hoje "acerta"
  o doc errado). A correção alinha o rótulo à realidade e a métrica passa a medir o
  que importa. Baseline pool-10 sem rerank: 0.605 (38Q).
- #35/#38 são falhas de **qualidade do índice** (extração letra-a-letra em PDFs) —
  não são corrigíveis por golden nem por retriever; exigem re-extração/OCR (Fase 2).
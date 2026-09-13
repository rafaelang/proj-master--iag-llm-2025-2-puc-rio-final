# Diagnóstico das 5Q de retrieval residual — P3 residual (rota simples)

Complemento do P3 (`docs/v05b_p3_slm_ab.md`): as 5Q onde nem o frontier pro responde
com o top-5 foram diagnosticadas **por página** (não só por doc), usando o índice P1
(RAG_DIR, pool 60, OCR) e o golden v0.5b congelado.

## Método

Para cada uma das 5Q: (1) posição do doc esperado no ranking RRF (pool 60, após dedup);
(2) posição das **páginas de conteúdo real** (identificadas manualmente a partir do
texto extraído do doc) no top-100; (3) o que o top-5 efetivamente entrega.

## Resultado

| # | Estrato | Pergunta | Doc esperado | Pos. doc (pool 60) | Página de resposta no top-100 | Classe |
|---|---|---|---|---|---|---|
| 22 | rotineira | joins em SQL | `tdp_aula05_joins..._06.pdf` | **1** | p.5 (27), p.9 (26), p.15 (23) | **A** |
| 23 | rotineira | subqueries | `oficina_tdp_aula03_tdp_07.pdf` | **3** | p.10 (39) | **A** |
| 24 | rotineira | modelagem lógica/física | `tdp_aula02_modelagem..._02.pdf` | **1** | p.4 (17) | **A** |
| 31 | rotineira | CNN | `pai_aula05_redes_convolucionais.pdf` | **19** | **nenhuma** (p.38+ fora do 100) | **B** |
| 40 | composta | agentes de IA em projetos | `proj_aula05_agentes...pdf` | 24 | p.6 (24) | **FALSO POSITIVO** |

## Leitura

**Classe A — doc certo, página errada (3Q: #22 #23 #24).** O retriever encontra o doc
(pos 1–3), mas o top-5 entrega a **página-título/intro** ("Transformando Dados em
Percepção", "Para aprendermos os diversos tipos de JOINS") em vez da página de
conteúdo (definição de join p.5+, subqueries p.10, modelagem lógica p.4). O pro (e o
SLM) abstêm-se ao ver só títulos. **Não é falha de recall de doc, é falha de
**seleção de página** dentro do doc — o top-k vence o doc mas não a página de resposta.

**Classe B — doc certo fora do top-5 (1Q: #31).** O doc CNN está na pos 19 e as páginas
de conteúdo (p.38+ "O que é uma Convolução?") **nem entram no top-100**: o texto
extraído do slide é fragmentado ("Redes Convolucionais\nO que é uma Convolução?\nFiltros
que deslizam pela imagem"), e BM25/denso não casam "rede neural convolucional (CNN)".
Falha de **matching léxico/semântico** sobre texto de slide mal extraído.

**Falso positivo (1Q: #40).** O SLM responde corretamente nas 4 variantes do P3
(juiz True); o "residual" veio de **erro transiente de API** do pro (`_gerar_api`
retornou vazio), não de retrieval. **Retirar #40 do residual.**

## Implicação para correção (P1-residual futuro)

1. **Seleção de página (Classe A):** o remédio de retriever não é pool maior (já 60)
   nem rerank genérico (degrada composta, P1). Candidatos: **dedup intra-doc por
   conteúdo** (evitar 2 páginas-título do mesmo doc no top-5), penalizar/descartar os
   **27 chunks boilerplate** ("Transformando Dados em Percepção", presentes em 8 docs)
   que roubam slots, e/ou rerank **só na rota simples** das 34Q (onde não há composta
   pesada). Impacto medível: recall por **página de resposta**, não por doc.
2. **Classe B (#31):** melhorar extração de slides (separar quebras de linha em frases)
   ou adicionar o termo "CNN" ao índice; reavaliar após P1.
3. **Métrica:** o recall@5 atual é por doc; para essas 3Q o recall@5=1 (doc entra) mas a
   resposta não é respondível. A métrica que importa para a v0.6 é **recall@5 da página
   de resposta** (ou taxa de respondibilidade com o pro).

Este diagnóstico **não altera** golden (congelado) nem a decisão do P3 (produção =
prompt v04, sem LoRA).
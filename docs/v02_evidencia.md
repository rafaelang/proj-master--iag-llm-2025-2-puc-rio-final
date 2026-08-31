# v0.2 - Evidencia - RAG com citacao e abstencao

- **Perguntas avaliadas:** 20
- **recall@5 (sobre perguntas com docs esperados):** 1.000
- **Taxa de abstencao correta (negativas + adversariais):** 1.000
- **Citacao presente (sobre respostas nao-abstidas):** 1.000
- **Top-k usado:** 5

## Metricas por estrato

| Estrato | n | recall@5 | abstencao correta | citacao presente |
|---|---|---|---|---|
| rotineira | 5 | 1.000 | - | 1.000 |
| composta | 5 | 1.000 | - | 1.000 |
| negativa | 5 | 0.000 | 1.000 | 0.000 |
| adversarial | 5 | 0.000 | 1.000 | 0.000 |

## Detalhes por pergunta

### 1. (rotineira) O que é RAG?
- **Resposta:** RAG, ou Retrieval-Augmented Generation, é uma técnica que conecta a resposta de um modelo de linguagem a documentos recuperados no momento da tarefa, substituindo a memória implícita do modelo por evidências rastreáveis e relevantes [4] proj_aula04_multiagentes_slm, pagina 23. O processo funciona extraindo embeddings de dados como PDFs e crawlers, recuperando os parágrafos mais similares à query e usando-os no prompt para gerar a resposta [7] nlp_aula06_rag_avancado_ocr, pagina 4. No LangChain, a memória baseada em vetor é apenas busca semântica, sem uma memória "viva", pois o vectorstore decide o que recuperar com base na similaridade dos embeddings [3] nlp_aula06_rag_avancado_ocr, pagina 25.
- **Abstencao:** False
- **recall@5:** True
- **Citacoes:** [(4, 'proj_aula04_multiagentes_slm', 23), (7, 'nlp_aula06_rag_avancado_ocr', 4), (3, 'nlp_aula06_rag_avancado_ocr', 25)]

### 2. (rotineira) O que é um embedding?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** True
- **Citacoes:** []

### 3. (rotineira) O que é fine-tuning?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** True
- **Citacoes:** []

### 4. (rotineira) O que é FastAPI?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** True
- **Citacoes:** []

### 5. (rotineira) O que é um modelo de difusão?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** True
- **Citacoes:** []

### 6. (composta) Como RAG e embeddings se relacionam?
- **Resposta:** RAG e embeddings se relacionam porque os embeddings são usados para recuperar informações relevantes: os dados são convertidos em embeddings e a query também passa por um extrator de embedding, permitindo buscar parágrafos mais similares à query por vizinhança de embeddings, que são então usados na geração da resposta [2] nlp_aula06_rag_avancado_ocr, pagina 4 e [3] nlp_aula06_rag_avancado_ocr, pagina 5.
- **Abstencao:** False
- **recall@5:** True
- **Citacoes:** [(2, 'nlp_aula06_rag_avancado_ocr', 4), (3, 'nlp_aula06_rag_avancado_ocr', 5)]

### 7. (composta) Como agentes e SLM podem ser usados juntos?
- **Resposta:** Agentes e SLMs podem ser usados juntos de forma que SLMs especializados realizem tarefas repetitivas e de alto volume, enquanto modelos maiores ficam reservados para exceções e tarefas complexas. Em aplicações agentic, o trabalho é decomposto em papéis estreitos como classificar intenção, extrair campos, rotear fluxos e validar formatos, o que reduz a necessidade de uma capacidade geral em cada chamada e torna os SLMs atraentes por economia operacional, com menor custo por chamada, menor latência e maior previsibilidade, desde que o sistema encaminhe exceções para capacidades mais fortes [1] proj_aula04_multiagentes_slm, pagina 3; [3] proj_aula04_multiagentes_slm, pagina 7. O melhor desenho costuma ser heterogêneo, com SLMs para volume, modelos maiores para exceções e pessoas para decisões que exigem responsabilidade, como em um fluxo de atendimento onde um SLM classifica a intenção e responde casos padrão, um modelo maior lida com ambiguidade e contexto complexo, e revisão humana é preservada para riscos altos ou ações irreversíveis [4] proj_aula04_multiagentes_slm, pagina 35; [10] proj_aula04_multiagentes_slm, pagina 19. Evoluções como o agentic RAG também mostram agentes autônomos que planejam, quebram perguntas complexas e decidem quando buscar mais informações, integrando SLMs em etapas especializadas [2] nlp_aula06_rag_avancado_ocr, pagina 30; [6] proj_aula04_multiagentes_slm, pagina 1.
- **Abstencao:** False
- **recall@5:** True
- **Citacoes:** [(1, 'proj_aula04_multiagentes_slm', 3), (3, 'proj_aula04_multiagentes_slm', 7), (4, 'proj_aula04_multiagentes_slm', 35), (10, 'proj_aula04_multiagentes_slm', 19), (2, 'nlp_aula06_rag_avancado_ocr', 30), (6, 'proj_aula04_multiagentes_slm', 1)]

### 8. (composta) Qual a relação entre banco de dados relacional e SQL?
- **Resposta:** A SQL é a linguagem padrão de consultas aos bancos de dados relacionais, e seus comandos são agrupados em cinco categorias: DDL, DML, DCL, DTL e DQL [1] tdp_aula03_sql_ddl, pagina 4. O curso também aborda a SQL em contextos específicos, como a criação de estruturas (DDL) e a manipulação de dados (DML) [2] tdp_aula03_sql_ddl, pagina 2.
- **Abstencao:** False
- **recall@5:** True
- **Citacoes:** [(1, 'tdp_aula03_sql_ddl', 4), (2, 'tdp_aula03_sql_ddl', 2)]

### 9. (composta) Como prompt engineering e RAG se complementam?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** True
- **Citacoes:** []

### 10. (composta) Como GAN e VAE diferem na geração de dados?
- **Resposta:** A GAN gera dados realistas imitando distribuições complexas, mas sem controle sobre o que é gerado, pois seu espaço de ruído é opaco e desorganizado [6] dgl_aula01_gan_vae, pagina 23. Já o VAE aprende um espaço latente estruturado, organizando os dados e permitindo reconstrução, combinando realismo com estrutura [5] dgl_aula01_gan_vae, pagina 21. Enquanto a GAN foca em replicar uma distribuição de probabilidade com uma função custo que mede a distância entre dados gerados e reais [2] dgl_aula01_gan_vae, pagina 15, o VAE é treinado com forças de reconstrução (erro pixel a pixel) e organização (KL divergence para proximidade de uma gaussiana padrão) [4] dgl_aula01_gan_vae, pagina 31.
- **Abstencao:** False
- **recall@5:** True
- **Citacoes:** [(6, 'dgl_aula01_gan_vae', 23), (5, 'dgl_aula01_gan_vae', 21), (2, 'dgl_aula01_gan_vae', 15), (4, 'dgl_aula01_gan_vae', 31)]

### 11. (negativa) Quem foi o primeiro presidente do Brasil?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []

### 12. (negativa) Qual a capital da França?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []

### 13. (negativa) Como fazer bolo de chocolate?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []

### 14. (negativa) Quem descobriu a América?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []

### 15. (negativa) O que é a teoria da relatividade?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []

### 16. (adversarial) O que é FastAPI no contexto de banco de dados relacional?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []

### 17. (adversarial) Como usar RAG para treinar uma GAN?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []

### 18. (adversarial) Qual a relação entre SQL e embeddings?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []

### 19. (adversarial) O que é chunking em modelos de difusão?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []

### 20. (adversarial) Como agentes multiagentes usam SQL?
- **Resposta:** NAO_SEI
- **Abstencao:** True
- **recall@5:** False
- **Citacoes:** []


# v0.2 - Evidencia - RAG (dataset unico do projeto2)

> Pipeline alvo: `projeto_final (BM25 + fastembed + RRF, DeepSeek deepseek-chat)`
> Dataset: `data/golden_set/rag/perguntas.json`.
> Criterios: projeto2 (detectar_abstencao, extrair_citacoes, auditar_citacao, juiz LLM-as-judge).
> Juiz: LLM-as-judge `deepseek-v4-pro` (mesmo prompt do 'juiz_correcao.py' v0.6).

## Metricas por estrato

| Estrato | n | recall@5 | acerto | abstencao correta | citacao presente |
|---|---|---|---|---|---|
| rotineira | 10 | 0.900 | 0.500 | 0.600 | 1.000 |
| composta | 4 | 0.750 | 0.250 | 0.250 | 1.000 |
| negativa | 2 | 0.500 | 1.000 | 1.000 | 1.000 |
| adversarial | 4 | - | 1.000 | 1.000 | 0.000 |

## Resumo honesto

- **recall@5** (sobre as 16 com `docs_esperados`): **0.812**
- **acurácia end-to-end** (sobre 20, juiz LLM-as-judge): **0.600**
- **abstenção correta** (sobre 20): **0.650**
- **abstenção indevida** (sobre as 16 com `deve_abster`: false): **0.438** (7)
- **citação presente** (respostas não-abstidas com `[n]`): **1.000**
- **auditoria de citação** — fiel: 7 · fora: 2 · fantasma: 0
- **alucinações** (juiz flag "alucinou"): 0
- juiz não julgou (erro de parse): 0

## Falhas detectadas

### #01 [rotineira] O que é RAG?
- **causa:** citacao fora
- **resposta:** RAG, ou Retrieval-Augmented Generation, é uma técnica que conecta a resposta de um modelo de linguagem a documentos recuperados no momento da tarefa, substituindo a memória implícita por evidências ra

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **causa:** resposta incorreta (juiz): A resposta apenas cita a fonte e a página, sem explicar como o FastAPI é usado para servir um modelo, não atendendo à pergunta.
- **resposta:** [1] pai_aula07_deploy_fastapi, pagina 32

### #06 [rotineira] O que é OCR e como ele ajuda na recuperação de documentos?
- **causa:** citacao fora
- **resposta:** OCR, ou Optical Character Recognition, é uma tecnologia que converte imagens de texto em dados de texto editáveis e pesquisáveis, sendo essencial para PDFs escaneados e fotografias de documentos físic

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **causa:** recuperacao falhou (nenhum doc esperado no top-5)
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve (NAO_SEI) mesmo havendo fonte esperada no corpus que poderia embasar a definição de fine-tuning.
- **resposta:** NAO_SEI

### #08 [rotineira] O que é transfer learning?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve mesmo havendo fonte esperada no corpus que deveria embasar o tema, portanto está incorreta.
- **resposta:** NAO_SEI

### #09 [rotineira] O que é WER e como ele é usado na avaliação de voz?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): Havia fonte esperada (proj_spec_tcc.pdf) que poderia embasar a resposta, mas o assistente se absteve com NAO_SEI, o que é incorreto neste ca
- **resposta:** NAO_SEI

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): Havia fonte esperada sobre o tema, mas o assistente se absteve com NAO_SEI, deixando de responder uma pergunta coberta pelo corpus.
- **resposta:** NAO_SEI

### #11 [composta] O que são embeddings e como são usados no RAG?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve (NAO_SEI) mesmo havendo fonte esperada que aborda o tema de embeddings e RAG.
- **resposta:** NAO_SEI

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **causa:** recuperacao falhou (nenhum doc esperado no top-5)
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve com NAO_SEI, mas existem fontes esperadas sobre prompt engineering que poderiam embasar a diferença em relação a fine-
- **resposta:** NAO_SEI

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve (NAO_SEI) mesmo havendo uma fonte esperada no corpus do curso, portanto não responde à pergunta.
- **resposta:** NAO_SEI

### #16 [negativa] O WER é uma métrica de qual modalidade: voz, imagem ou banco de dados?
- **causa:** recuperacao falhou (nenhum doc esperado no top-5)
- **resposta:** O WER é uma métrica da modalidade de voz, pois é mencionado no contexto de um protótipo de voz funcional, onde se mede o WER antes e depois do vocabulário customizado. [7] proj_spec_tcc, pagina 1

## Detalhes por pergunta

### #01 [rotineira] O que é RAG?
- **Resposta:** RAG, ou Retrieval-Augmented Generation, é uma técnica que conecta a resposta de um modelo de linguagem a documentos recuperados no momento da tarefa, substituindo a memória implícita por evidências rastreáveis e relevantes [4] proj_aula04_multiagentes_slm, pagina 23. O processo funciona com dados, um extrator de embeddings, recuperação da informação, prompt engineering e geração de respostas, onde parágrafos mais similares à query são recuperados por vizinhança de embeddings [7] nlp_aula06_rag_avancado_ocr, pagina 4.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define RAG corretamente e descreve o processo com embeddings, recuperação e geração, alinhada ao conteúdo da disciplina.
- **Citações:** [4] proj_aula04_multiagentes_slm.pdf; [7] nlp_aula06_rag_avancado_ocr.pdf

### #02 [rotineira] O que significa DDL e para que serve?
- **Resposta:** DDL significa Linguagem de Definição de Dados, e serve para criar e modificar objetos em um banco de dados, como tabelas, índices e visões [1] tdp_aula03_sql_ddl, pagina 5. O comando Create é usado para criar esses objetos, enquanto o comando Alter é utilizado para realizar modificações em tabelas [7] tdp_aula03_sql_ddl, pagina 15.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define DDL corretamente como Linguagem de Definição de Dados e explica seus comandos Create e Alter para criar e modificar objetos, conforme o material da aula.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf; [7] tdp_aula03_sql_ddl.pdf

### #03 [rotineira] O que é DQL?
- **Resposta:** DQL é a Linguagem de Consultas de Dados, uma das cinco categorias de comandos da SQL, a linguagem padrão de consultas aos bancos de dados relacionais [1] tdp_aula03_sql_ddl, pagina 4.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente DQL como Linguagem de Consultas de Dados, uma das cinco categorias de comandos SQL, conforme o material da aula.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **Resposta:** [1] pai_aula07_deploy_fastapi, pagina 32
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 2
- **Justificativa do juiz:** A resposta apenas cita a fonte e a página, sem explicar como o FastAPI é usado para servir um modelo, não atendendo à pergunta.
- **Citações:** [1] pai_aula07_deploy_fastapi.pdf

### #05 [rotineira] O que são GAN e VAE?
- **Resposta:** GAN e VAE são modelos generativos de IA. A GAN é um jogo entre dois modelos em competição: um gerador que aprende a criar dados falsos e um discriminador que aprende a distinguir real de falso, e o equilíbrio entre eles leva à qualidade [1] dgl_aula01_gan_vae, pagina 11. A GAN gera imagens realistas, mas sem controle, pois o espaço de ruído é opaco e desorganizado [5] dgl_aula01_gan_vae, pagina 23. Já o VAE é um modelo que combina compressão com estrutura probabilística, aprendendo a organizar o mundo, pois ele comprime, organiza e reconstrói [4] dgl_aula01_gan_vae, pagina 26. O VAE funciona como um funil inteligente que comprime a imagem em sua essência no espaço latente e reconstrói a partir dessa representação [3] dgl_aula01_gan_vae, pagina 28, e o encoder mapeia a imagem para uma distribuição gaussiana, não um ponto fixo [9] dgl_aula01_gan_vae, pagina 30. Enquanto a GAN gera bem sem controle, o VAE aprende um espaço estruturado [7] dgl_aula01_gan_vae, pagina 21.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta descreve corretamente GAN e VAE conforme o material indicado, contrastando geração realista sem controle da GAN com o espaço latente estruturado do VAE.
- **Citações:** [1] dgl_aula01_gan_vae.pdf; [5] dgl_aula01_gan_vae.pdf; [4] dgl_aula01_gan_vae.pdf; [3] dgl_aula01_gan_vae.pdf; [9] dgl_aula01_gan_vae.pdf; [7] dgl_aula01_gan_vae.pdf

### #06 [rotineira] O que é OCR e como ele ajuda na recuperação de documentos?
- **Resposta:** OCR, ou Optical Character Recognition, é uma tecnologia que converte imagens de texto em dados de texto editáveis e pesquisáveis, sendo essencial para PDFs escaneados e fotografias de documentos físicos quando a extração direta retorna vazio [1] nlp_aula06_rag_avancado_ocr, pagina 8. Ele ajuda na recuperação de documentos ao transformar conteúdo baseado em imagem em texto utilizável, o que é crucial para o pipeline de RAG, pois permite que a extração de dados funcione mesmo quando o PDF não possui texto nativo [2] proj_aula04_multiagentes_slm, pagina 23. Bibliotecas como PyPDFLoader, que funciona bem para PDFs nativos, e Tesseract, um motor de OCR maduro, são usadas nesse processo [5] nlp_aula06_rag_avancado_ocr, pagina 7.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define OCR corretamente e explica sua importância na recuperação de documentos para RAG, com suporte na fonte esperada (nlp_aula06). A citação de fonte extra não invalida o conteúdo, pois está alinhado ao tema do curso.
- **Citações:** [1] nlp_aula06_rag_avancado_ocr.pdf; [2] proj_aula04_multiagentes_slm.pdf; [5] nlp_aula06_rag_avancado_ocr.pdf

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 0.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve (NAO_SEI) mesmo havendo fonte esperada no corpus que poderia embasar a definição de fine-tuning.
- **Citações:** -

### #08 [rotineira] O que é transfer learning?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve mesmo havendo fonte esperada no corpus que deveria embasar o tema, portanto está incorreta.
- **Citações:** -

### #09 [rotineira] O que é WER e como ele é usado na avaliação de voz?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** Havia fonte esperada (proj_spec_tcc.pdf) que poderia embasar a resposta, mas o assistente se absteve com NAO_SEI, o que é incorreto neste caso.
- **Citações:** -

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** Havia fonte esperada sobre o tema, mas o assistente se absteve com NAO_SEI, deixando de responder uma pergunta coberta pelo corpus.
- **Citações:** -

### #11 [composta] O que são embeddings e como são usados no RAG?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve (NAO_SEI) mesmo havendo fonte esperada que aborda o tema de embeddings e RAG.
- **Citações:** -

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 0.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve com NAO_SEI, mas existem fontes esperadas sobre prompt engineering que poderiam embasar a diferença em relação a fine-tuning, portanto está incorreta.
- **Citações:** -

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve (NAO_SEI) mesmo havendo uma fonte esperada no corpus do curso, portanto não responde à pergunta.
- **Citações:** -

### #14 [composta] Como Whisper e WER se relacionam com o protótipo de voz do projeto?
- **Resposta:** O Whisper é citado como uma das tecnologias para o pipeline de fala do projeto, que inclui ASR, diarização e TTS, e o WER é uma métrica que deve ser medida no protótipo de voz. Especificamente, a entrega v0.1 do projeto exige um protótipo de voz funcional, com WER medido em 10 amostras próprias, antes e depois do vocabulário customizado, que serve para corrigir problemas como ruído de ambiente, sotaque e jargão técnico. Portanto, o Whisper é uma tecnologia usada no pipeline de fala, e o WER é a métrica para avaliar a qualidade da transcrição do protótipo. [1] proj_spec_tcc, pagina 1
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta descreve corretamente que Whisper é tecnologia de ASR no pipeline de fala e que WER é métrica avaliada no protótipo, incluindo detalhes de v0.1, amostras e customização, alinhados ao documento do projeto.
- **Citações:** [1] proj_spec_tcc.pdf

### #15 [negativa] Qual destes termos NÃO está relacionado a bancos de dados: DDL, DQL ou GAN?
- **Resposta:** GAN não está relacionado a bancos de dados, pois DDL e DQL são categorias da linguagem SQL, conforme [1] tdp_aula03_sql_ddl, pagina 4.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente GAN como não relacionado a bancos de dados e cita a fonte do curso para DDL e DQL.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf

### #16 [negativa] O WER é uma métrica de qual modalidade: voz, imagem ou banco de dados?
- **Resposta:** O WER é uma métrica da modalidade de voz, pois é mencionado no contexto de um protótipo de voz funcional, onde se mede o WER antes e depois do vocabulário customizado. [7] proj_spec_tcc, pagina 1
- **Absteve:** False · **Deve abster:** False · **recall@5:** 0.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente WER como métrica de voz, alinhada ao contexto do documento sobre protótipo de voz funcional.
- **Citações:** [7] proj_spec_tcc.pdf

### #17 [adversarial] Qual a previsão do tempo para amanhã no Rio de Janeiro?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** Pergunta fora do corpus do curso e o assistente se absteve corretamente com NAO_SEI.
- **Citações:** -

### #18 [adversarial] Quem ganhou a Copa do Mundo de 2022?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus do curso e o assistente se absteve corretamente, sem alucinar.
- **Citações:** -

### #19 [adversarial] Explique a teoria da relatividade geral de Einstein.
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** Pergunta fora do corpus do curso e o assistente se absteve, conforme esperado.
- **Citações:** -

### #20 [adversarial] Qual o preço atual da ação da Petrobras?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus e a resposta se absteve corretamente com NAO_SEI.
- **Citações:** -


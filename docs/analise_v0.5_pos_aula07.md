# Análise pós-aula07 — Revisão da v0.5 (Adaptação) à luz do conteúdo sobre SLM

> Objetivo: revisar as decisões e experimentos da **v0.5 · Adaptação** do `projeto2`
> (ver `analise_v0.5.md`) usando como régua o conteúdo de **AULAS/PROJ/aula07** (SLM em
> Produção e alinhamento por preferência). Para cada ponto, a referência à aula07 é
> indicada entre parênteses — `<arquivo>`: slide/bloco/ato.

---

## 0. Referências da aula07 citadas neste documento

| Recurso | Como é citado |
|---|---|
| `SLM_em_Produção.pptx` | `slide N` |
| `SLMmodelo_pequeno.ipynb` | `lab, bloco N` |
| `rlhf_academia_desculpas.ipynb` | `rlhf, ato N` |
| `rlaif_constituicao_e_auditoria.ipynb` | `rlaif, ato N` |
| `SLM_manipulando.ipynb` | `dpo, bloco N` |
| `AULAS/PROJ/aula07/documentacao.md` | `doc aula07` |

---

## 1. Resumo da v0.5 (o que foi decidido e medido)

- **Decisão central:** manter **prompt + RAG**; **não vale a pena treinar (nem LoRA)**,
  defendido por: requisito é de grounding (citação + abstenção), corpus pequeno (10 docs),
  falta de pares Q&A rotulados e custo de manutenção.
- **Mini-experimento (8 perguntas):** abstenção prompt-only **0.75** × RAG **1.00**;
  citação [n]: 0/8 × 4/8.
- **Matriz 2×2 (20 perguntas, Qwen2.5-1.5B):** sem RAG base 0.70 / destilado 0.65; com RAG
  base **0.75** / destilado **0.95**. Conclusão: destilar + RAG agrega (32,6 s de GPU na T4)
  e **empata com o multiagente (router + pro) da v0.4 a custo zero de API**.
- **Extra DPO (24 pares, preferido=com-RAG × rejeitado=sem-RAG, "tom de tutor"):** pipeline
  validado, efeito marginal (0.95 → 0.90); LLM-as-judge de tom: SFT 1.89 × SFT+DPO 1.89.
- **Falha metodológica reconhecida:** a métrica de abstenção não mede acurácia (recompensou
  alucinação no caso #15) → veredito de correção delegado à v0.6, que mostrou: base 0.45,
  sft 0.21, sft_dpo 0.40 (acurácia); a destilação **piorou** a acurácia real.

---

## 2. Revisão da decisão "não treinar" frente à aula07

### 2.1. A sequência da v0.5 é exatamente a "escada de intervenção" da aula

A aula é explícita: **esgote prompt → RAG → formato de saída antes de mexer nos pesos**
(`SLM_em_Produção.pptx, slide 17`; reforçado na escada do `slide 18`: Prompt → RAG →
LoRA/QLoRA → treino completo, "avance apenas quando o degrau anterior for exaurido"). A v0.5
fez exatamente isso:

1. Prompt-only falhou no requisito (abstenção 0.75, alucinou Copa 2022/relatividade) —
   degrau 1 **exaurido e reprovado**;
2. RAG cumpriu o requisito (abstenção 1.00, citação [n]) — degrau 2 **atende**;
3. LoRA testado como degrau 3 **superior** (destilado+RAG 0.95) — mas com ressalvas que a
   própria aula antecipa (ver §2.2 e §4).

### 2.2. O checklist go/no-go de treino (7 perguntas) confirma a decisão

O `slide 32` traz um checklist: *"Menos de cinco respostas 'sim' significa não treinar ainda."*
Aplicado à v0.5:

| Pergunta do slide 32 | Situação v0.5 | Sim? |
|---|---|---|
| Conjunto de avaliação separado, métrica + baseline medido? | Golden set 20 perguntas, métrica apenas de abstenção | ⚠️ parcial |
| Prompt e RAG exauridos? | Prompt reprovado; RAG não explorado a fundo (top-k, rerank) | ⚠️ parcial |
| Falha é de **FORMA** (formato/tom/terminologia) e não de **FATO**? | A falha é de **grounding** (fato/citação) — não de forma | ❌ |
| 200+ exemplos na distribuição real, rótulo revisado por humano? | 50 Q&A **sintéticos**, sem curadoria humana | ❌ |
| Escopo congelado por meses? | Corpus do curso muda a cada aula | ❌ |
| Há dono nomeado do adaptador/serviço? | Não | ❌ |
| Ganho proj. paga o imposto operacional? | Não medido (sem TCO) | ❌ |

**Score: 0–1 "sim"** → o checklist da aula **reforça a decisão v0.5 de não treinar**, agora
com critério explícito. O ponto mais forte é o terceiro: a aula diz que **treinar corrige
forma, não fato** (`slide 17`: "Treinamento não cria fatos novos de forma confiável") — e o
requisito do projeto é exatamente de fato (citar fonte e abster fora do corpus).

### 2.3. Anti-gatilhos da v0.5: volume, eval e qualidade

O `slide 15` lista cinco sinais vermelhos; três incidem aqui:

- **Volume baixo:** "Abaixo de ~1 milhão de tokens/mês, a economia não paga a engenharia.
  Use API e siga a vida." A v0.5 é um protótipo acadêmico individual, volume baixíssimo —
  o **anti-gatilho n° 1 da aula**. O cenário do `slide 38` ("Startup em validação": <1M
  tokens/mês → "API frontier, e ponto"; treinar? "Não") é o retrato do projeto → *"A
  primeira pergunta do seu relatório é: em qual coluna o meu caso está?"* — a resposta do
  caso v0.5 é a coluna "não treinar".
- **Ausência de eval:** "Sem conjunto de avaliação, você não sabe se piorou. E vai
  descobrir em produção, pelo cliente" — exatamente o que aconteceu na v0.5/v0.6: a métrica
  só de abstenção **escondeu** que a destilação piorou a acurácia (0.45 → 0.21 na v0.6).
- **Qualidade como produto** (`slide 15` e `slide 11`): respostas que o cliente lê/julga e
  tarefas abertas/subjetivas são território do frontier. O requisito "citação + abstenção"
  é verificável e objetivo — pertence ao RAG/SLM — **mas a síntese final para o usuário**
  (aula, `slide 12`: "Síntese Final — gerar o texto final que o cliente vai ler e julgar")
  é chamada frontier. A v0.5 não separou esses dois papéis ao avaliar o SLM.

### 2.4. Síntese desta seção

**Revisão:** a decisão "não treinar" foi **correta e agora está fundamentada pela aula07**:
a v0.5 até foi "precoce" no LoRA (degrau 3 foi testado com dataset sem curadoria e métrica
sem acurácia), mas a conclusão final (não treinar por ora) coincide com a régua da aula —
anti-gatilho de volume, falha de fato (não de forma), sem eval adequado, sem TCO. A aula
inclusive valida que *"reconhecer um anti-gatilho e escolher NÃO usar SLM é uma decisão
técnica válida"* (`slide 15`).

---

## 3. Revisão da destilação (LoRA) à luz da aula07

### 3.1. O lab seguiu o passo 1 da aula, mas pulou a curadoria

A aula define destilação como 4 passos (`slide 20`): **1. Coleta** de saídas do frontier na
tarefa real → **2. Curadoria** ("manter somente o que um humano assinaria embaixo") →
**3. Treino** LoRA → **4. Troca** (servir SLM; frontier só como escalada).

A v0.5 executou: 50 Q&A gerados pelo DeepSeek pro a partir de 8 docs → LoRA 4-bit
(r=8, α=16, 2,18M params ≈ 0,14%, 39 steps, 32,6 s T4). **Ou seja: passos 1 e 3 sim;
passo 2 (curadoria humana) não.** A aula é taxativa: *"Erro de rótulo vira comportamento
aprendido"* (`slide 19`) e *"o modelo grande gera, um humano precisa auditar a amostra"*
(`slide 19`, `doc aula07`).

**Consequência observada na própria v0.6:** o SFT destilado sem curadoria produziu **14
alucinações** (o dobro do base) — comportamento aprendido do dataset sintético ruidoso,
exatamente o risco apontado pelo slide 19.

### 3.2. "O conjunto de dados vale mais que o modelo"

O `slide 19` exige: distribuição real (inclusive entradas "feias"), rótulo revisado, casos
difíceis de propósito e gold set separado. A v0.5 tinha: dataset gerado a partir de trechos
**limpos** dos docs, sem casos de abstenção empírica, sem estratificação de dificuldade, e
eval em golden set separado (ok). Comparação com o checklist:

| Requisito (slide 19) | v0.5 | Pendência |
|---|---|---|
| Distribuição real (inclusive entradas ruins) | trechos limpos dos docs | faltam perguntas do estilo real (pessoa atrasada, pedido informal) |
| Rótulo revisado por humano | gerado por frontier, sem auditoria | **auditar amostra do dataset sintético** |
| Casos difíceis de propósito | adversariais existiam no golden set, mas não no dataset de treino | incluir casos de abstenção no treino |
| Gold set separado/congelado | golden set de 8–20 perguntas (pequeno) | expandir para 100–300 (`slide 49`) |

### 3.3. O teto do SLM (onde o frontier se afasta)

A aula é franca sobre o limite (`slide 11` = "O SLM continua tendo um teto"):
"Conhecimento amplo", "raciocínio novo/longo", "conversa geral", "contexto longo" são do
frontier. A única falha restante da v0.5/v0.6 (#19 — teoria da relatividade) é "limitação
genuína de um 1.5B" — a aula explicaria que **essa pergunta foge do teto** do SLM e deve ser
roteada como "caso ambíguo" (`slide 12`) ou escalada (`slide 13`). Nesse ponto, o projeto já
fez melhor que o SLM sozinho: a v0.4 roteia complexas para o frontier (cascata).

### 3.4. Síntese

**Revisão:** o lab provou viabilidade técnica, mas a aula07 mostra **por que o resultado
0.95 não se sustentaria com a régua certa**: sem curadoria, sem casos de abstenção no treino
e com métrica só de abstenção, o "ganho" medido era frágil — a v0.6 confirmou (SFT piora
acurácia). A recomendação não é descartar a destilação, é **elevá-la ao padrão da aula
antes de qualquer novo uso** (§6).

---

## 4. Revisão do DPO (extra da v0.5) à luz da aula07

### 4.1. "SFT é o ponto de parada padrão"; DPO exige preferências reais

O `slide 21` é direto: **SFT** é o primeiro passo (e, para a maioria, o único necessário);
**DPO** "exige preferências reais coletadas entre respostas (A é melhor que B). Não use se
as preferências forem inventadas ou sintéticas sem validação."

A v0.5 fez exatamente o que o slide **proíbe**: preferências = "preferido = com-RAG" ×
"rejeitado = sem-RAG", construídas por uma regra sintética. O `SLM_manipulando.ipynb`
(dpo) mostra que preferência alinha — mas com **30 pares de preferência real sobre um
estilo bem definido** (`dpo, bloco 1 e 5`). A v0.5 usou 24 pares com sinal **fraco e
circular** (RAG vs sem-RAG não é "preferência forte", é proxy de grounding) — resultado:
efeito marginal (0.95 → 0.90) e tom de tutor empatado (1.89 × 1.89).

**Interpretação à luz da aula:** o "achado honesto" da v0.5 é o resultado esperado quando se
viola o slide 21 — sinal de preferência sintético não puxa a política para lugar claro.

### 4.2. O juiz de tom precisava de auditoria (pedido da própria aula)

A v0.5 usou LLM-as-judge (DeepSeek pro) para medir "tom de tutor" com notas 1–5. A aula
`rlaif_constituicao_e_auditoria.ipynb` dedica o **ato 4** a auditar exatamente esse tipo de
juiz, com 3 perguntas que a v0.5 não fez:

1. **O juiz é consistente?** (inverter A/B muda o voto?) — não medido;
2. **Está julgando conteúdo ou tamanho?** (viés de comprimento) — não medido;
3. **Obedece à própria rubrica?** — não medido.

Para o projeto, reproduzir o ato 4 do `rlaif` no `tom_judge.json` da v0.5 custaria
centavos e responderia se o empate 1.89×1.89 é real ou artefato do juiz. A conclusão do
`rlaif` ("Se o juiz falhar em 1 ou 3, o problema não se resolve com mais rótulos: mais
rótulos só multiplicam o mesmo erro") é diretamente aplicável.

### 4.3. Reward hacking — analogia direta com o vazamento do sinal

O `rlhf` (`ato 5`) mostra que o RM aprende a **correlação** e a política explora o atalho.
Na v0.5, o proxy "com-RAG = preferido" é um atalho parecido: a política pode "aprender a
amar a evidência" sem melhorar o comportamento real — e a v0.6 mostrou que realmente não
melhorou (DPO 0.40 < base 0.45). A lição da aula: **o objetivo não é maximizar o sinal
(RM/juiz), é não degradar o produto** (`rlhf, ato 4`: recompensa subindo sozinha não quer
dizer nada; as 3 colunas juntas são o painel).

### 4.4. Síntese

**Revisão:** o experimento DPO da v0.5 foi um exercício válido de pipeline, mas **fora da
especificação da aula para usar DPO** (`slide 21`: preferências reais; `rlaif, ato 4`:
juiz auditado). O resultado "marginal" era o esperado. Manter como **demonstrativo** — e só
reativar com preferências reais (§6).

---

## 5. Lacunas que a aula07 expõe na v0.5

1. **Sem métrica de acurácia/ancoragem** — a aula exige "medir com número na tarefa real"
   (`lab, bloco 5`; `slide 32`). A v0.5 mediu só abstenção; o requisito (citação) ficou
   sem métrica de correção até a v0.6.
2. **Sem TCO / planilha de custo** — o `slide 33–36` define TCO mensal =
   `Inferência + GPU + Operação + Engenharia + Retrabalho + Escaladas`, com "premissas
   visíveis" e ponto de equilíbrio por volume (`slide 36`). A v0.5 argumentou "US$ 5–30 de
   GPU" sem plano de custo completo e **não mediu o retrabalho/escalada** — o custo que
   "silenciosamente estoura o projeto" (`slide 34`).
3. **Gold set pequeno (8–20 perguntas)** — a aula recomenda **100–300 casos**, amostragem
   honesta com fáceis/medianos/difíceis, **critério escrito** e **congelamento** antes de
   ajustar (`slide 49` e `slide 19`). A v0.5 usou golden set de 20 sem critério formalizado
   por escrito.
4. **Juiz não auditado** — ver §4.2 (`rlaif, ato 4`).
5. **Escopo tratado como estável quando não é** — a aula pede escopo congelado para treinar
   (`slide 32`); o corpus do curso **muda a cada aula** (novas releases do próprio
   projeto) — mais um anticorpo contra treinar agora.
6. **Temperatura/geração não padronizada** — o `lab, bloco 6` é claro: "tarefa com resposta
   certa pede temperatura 0". A v0.5 re-gerou respostas truncadas (150 chars) e só a v0.6
   padronizou; para medir corretamente, fixar seed + `temperature=0` (`lab, bloco 5–6`).

---

## 6. Recomendações (revisões concretas sobre a v0.5)

### R1. Confirmar e documentar "não treinar" com a régua da aula
Registrar no relatório da release: checklist do `slide 32` (score 0–1 "sim"), anti-gatilho de
volume (`slide 15`/`slide 38`, cenário "startup/validação"), e a regra "treinar corrige
forma, não fato" (`slide 17`). Isso transforma a decisão de *intuição* em *decisão
defendida com critério explícito* — que é o que a aula pede ("defendida com número — não
com opinião", `slide 1`).

### R2. Expandir e formalizar o golden set
Crescer para **100–300 casos** (`slide 49`) com: estratos (rotineiras, compostas, negativas,
adversariais — já usados na v0.6), **critério de rótulo escrito** (duas pessoas concordam)
e **congelamento** antes de qualquer ajuste. Congelar também `temperature=0` e seed
(`lab, bloco 6`). Este gold set passa a ser o "contrato de avaliação" das próximas releases.

### R3. Métrica dupla: abstenção + correção/ancoragem (com juiz auditado)
Mantendo o juiz de correção da v0.6, **auditar o juiz** com o protocolo do `rlaif, ato 4`
(consistência na inversão A/B, viés de comprimento, obediência à rubrica) e **calibrar com
amostra humana** (IA rotula, humano audita ~200 sorteados; concordância < 60% = alarme —
`rlaif, doc aula07`). O dossel "juiz de IA escala; gold set humano continua a referência"
(`slide 30`) vira política do projeto.

### R4. Só retreinar se a falha for de FORMA
Deixar explícito o teste do `slide 16`: "A falha do modelo é de forma ou de fato?" Se de
**forma** (formato/tom/terminologia) → avaliar LoRA/QLoRA (`slide 18`); se de **fato**
(caso atual, v0.6 evidenciou: SFT piorou acurácia) → **melhorar o RAG, não os pesos**:
top-k, rerank, chunking (a v0.2/v0.3 já avançaram isso). Prioridade pós-aula07: aperfeiçoar
à esquerda da escada (`slide 18`), não subir de degrau.

### R5. Destilação "à moda da aula", se/quando revisitada
Se um dia quiser recuperar o 0.95 (destilado+RAG) de forma sustentável, seguir o
`slide 20` completo:
1. coletar saídas do frontier na **tarefa real de produção** (não em trechos limpos);
2. **curadoria humana** do dataset (podar alucinações como as 14 da v0.6);
3. LoRA sobre Qwen2.5-1.5B-4bit (setup já validado: 32,6 s T4) — aproveitar o adapter
   `slm_adapter/` como ponto de partida;
4. servir SLM + frontier só como escalada, com gold set R2 medindo acurácia+abstenção.
Incluir **casos de abstenção** no dataset de treino (o destilado "fica super-cauteloso sem
RAG" — v0.5) e casos difíceis (`slide 19`).

### R6. DPO: somente com preferências reais e juiz auditado
Suspender o sinal sintético "com-RAG × sem-RAG" (viola `slide 21`). Para reativar, coletar
preferências reais A>B (ex.: saídas de duas rotas do v0.4 com anotação humana/do aluno),
validar o DPO com a auditoria do `rlaif, ato 4` no juiz, e medir com o gold set R2 +
"3 colunas do painel" (`rlhf, ato 4`: recompensa, diversidade, verossimilhança contra o modelo
de referência) para detectar reward hacking cedo (`rlhf, ato 5`).

### R7. Montar a planilha de TCO (3 cenários)
Usar o `slide 33–36`/`slide 38` no relatório de produção (v1.0):
- colunas: API frontier × API de modelo pequeno × SLM hospedado/self-host;
- linhas do `slide 34`: inferência (incl. tokens de raciocínio), GPU/infra, operação(MLOps),
  engenharia, retrabalho, escaladas;
- premissas visíveis e ponto de equilíbrio (`slide 36`). Para o caso atual (volume baixo),
  a conclusão esperada coincide com a v0.5: a conta fecha para **não** treinar.
Em paralelo, **medir tokens de saída e escaladas reais** do pipeline v0.4 (`slide 37`:
o "barato" pode pensar demais).

### R8. Roteamento para modelos abertos dentro do SLM
A v0.5 mediu o SLM "sozinho" e "com RAG", mas a arquitetura ótima da aula é a **cascata
heterogênea** (`slide 12`, `slide 13`, `slide 41`). Recomendação: avaliar o SLM/local
apenas como **executor operacional** (extração, roteamento, validação de schema) e manter
a **síntese final** no frontier — já é o desenho da v0.4; falta **medir** essa separação no
gold set R2 (tráfego e custo por rota, como sugere a trilha C do `slide 48`).

### R9. Reportar "o que foi descartado" (resultado válido)
A aula valoriza conclusões negativas: "Um projeto que conclui 'medimos, comparamos e SLM não
compensa aqui' é um projeto bem-sucedido" (`slide 39`) e a entrega final inclui "defesa do
que foi medido, decidido e descartado" (`slide 50`). A v0.5 já é esse caso — formatar o
relatório final com: comparativo de 4 abordagens (já existe), checklist go/no-go, TCO,
matriz 2×2 e os resultados da v0.6 como evidência.

---

## 7. Conclusão

A aula07 **confirma a decisão central da v0.5** (não treinar; manter RAG) e dá a ela um
vocabulário e critérios que faltavam: anti-gatilhos de volume/eval/formato (`slide 15`),
checklist go/no-go (`slide 32`), escada de intervenção (`slide 18`) e TCO (`slide 33–36`).
Ao mesmo tempo, a aula expõe as **fragilidades de medição** da v0.5 — gold set pequeno,
métrica sem acurácia, dataset sintético sem curadoria, DPO com preferência sintética e juiz
não auditado — e prescreve como corrigi-las (gold set ≥100–300, juiz auditado conforme
`rlaif, ato 4`, destilação com curadoria conforme `slide 20`, DPO com preferências reais
conforme `slide 21`).

**Síntese operacional:** a v0.5 estava certa no destino ("não treinar agora") e apressada no
caminho (medir e treinar com rigor). As recomendações R1–R9 transformam a decisão em um
relatório defendido com número — que é o padrão de entrega que a própria aula07 exige
(`slide 1` e `slide 50`).
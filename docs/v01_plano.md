# Plano v0.1 - Voz (Prototipo funcional com FastAPI + HTML)

## Objetivo da release

Entregar um **prototipo funcional de chat por voz** acessivel via navegador:

1. Pagina HTML estatica servida pelo FastAPI com gravacao de audio pelo microfone.
2. Endpoint `/chat` que recebe o audio e executa o pipeline:
   - ASR: `faster-whisper` small com e sem prompt de vocabulario de dominio.
   - LLM: DeepSeek (via SDK OpenAI) para gerar a resposta textual.
   - TTS: `piper-tts` local para sintetizar a resposta em audio.
3. Avaliacao de WER em 10 amostras proprias, comparando transcrição **sem** e **com** vocabulario.
4. Logs de debug em todas as etapas usando `loguru`.

## Escopo

- Reimplementar do zero (consultando `projeto2/` como referencia, sem copiar codigo diretamente).
- Entrega da v0.1: pagina funcional + endpoint + avaliacao WER + documentacao.
- Nao incluir RAG (v0.2), visao (v0.3) nem agentes (v0.4).
- Resposta do LLM pode ser generica sobre o dominio do curso; nao e necessario citar fonte ainda.

## Dependencias

Adicionar ao `pyproject.toml` por release, sem bloat:

- `faster-whisper>=1.1` - ASR local (CPU).
- `piper-tts>=1.2` - TTS local em pt-BR.
- `python-multipart>=0.0.9` - upload de arquivo no endpoint `/chat`.
- `fastapi>=0.115` - framework da API.
- `uvicorn>=0.30` - servidor ASGI.
- `openai>=1.40` - SDK compativel com DeepSeek.
- `loguru>=0.7` - logging estruturado em todas as etapas.
- `pytest>=8.0` - testes.

## Estrutura de arquivos

```
projeto_final/
├── src/
│   └── projeto_final/
│       ├── __init__.py
│       ├── main.py              # FastAPI: serve HTML e endpoint /chat
│       ├── voz.py               # transcrição com faster-whisper + vocabulario
│       ├── tts.py               # sintese com piper-tts
│       ├── llm.py               # cliente DeepSeek via openai
│       ├── config.py            # carrega .env e expoe constantes
│       └── wer.py               # calculo de WER
├── prompts/
│   └── v0.1/
│       └── vocabulario_voz.txt  # termos do dominio como prompt-guia
├── data/
│   ├── raw/                     # nao versionado
│   ├── processed/               # nao versionado
│   └── golden_set/
│       └── voz/
│           ├── referencias.json # 10 frases de referencia
│           └── 01.ogg ... 10.ogg # 10 amostras de audio proprias
├── tests/
│   └── test_v01.py              # testes da v0.1
├── docs/
│   ├── v01_plano.md             # este arquivo
│   └── v01_voz_evidencia.md     # resultado da avaliacao
└── static/
    └── index.html               # pagina estatica do chat por voz
```

## Pipeline do endpoint `/chat`

1. Recebe `POST /chat` com `multipart/form-data` contendo o arquivo de audio.
2. Salva o audio temporariamente em `data/processed/temp/`.
3. Transcreve com `faster-whisper` small:
   - Modo padrao (sem vocabulario).
   - Modo com prompt de vocabulario (`prompts/v0.1/vocabulario_voz.txt`).
4. (Opcional v0.1) Envia o texto transcrito para DeepSeek com um prompt simples de assistente do curso.
5. Sintetiza a resposta com `piper-tts` para um arquivo `.wav`.
6. Retorna o audio `.wav` com headers customizados:
   - `X-Transcription`: texto transcrito.
   - `X-Answer`: resposta textual do LLM.
7. Todos os passos logados com `loguru` (debug).

## Pagina HTML (`static/index.html`)

Baseada no exemplo fornecido, mas mantendo **apenas a secao de resultados da v0.1** (inicio das avaliacoes). A tabela de WER deve:

- Listar os 10 arquivos (`01.ogg` ... `10.ogg`).
- Colunas: Arquivo, WER sem, WER com, Acuracia sem, Acuracia com.
- Linha de media ao final.
- Valores iniciais podem ser preenchidos apos a avaliacao; na primeira versao da pagina podem constar "a medir" ou zeros.

A secao da v0.2 (RAG) nao deve aparecer na pagina da v0.1.

## Avaliacao WER

1. Criar 10 frases de referencia sobre o dominio do curso em `data/golden_set/voz/referencias.json`.
2. Gravar ou gerar 10 arquivos de audio correspondentes (`01.ogg` ... `10.ogg`).
3. Rodar o script `src/projeto_final/wer.py` que:
   - Transcreve cada audio sem vocabulario.
   - Transcreve cada audio com vocabulario.
   - Calcula WER e acuracia (1 - WER) para cada par.
   - Gera a tabela de resultados.
4. Atualizar a pagina `static/index.html` com os valores medidos.
5. Documentar em `docs/v01_voz_evidencia.md`.

## Testes

- `tests/test_v01.py`:
  - Verifica se o endpoint `/chat` responde 200 com um audio de teste.
  - Verifica se os headers `X-Transcription` e `X-Answer` estao presentes.
  - Verifica se o calculo de WER esta correto para casos conhecidos.

## Criterios de aprovacao

- [ ] `uv sync` instala todas as dependencias sem erro.
- [ ] `pytest` passa.
- [ ] Servidor sobe com `uvicorn src.projeto_final.main:app --reload`.
- [ ] Pagina carrega em `/` e consegue gravar/enviar audio.
- [ ] Endpoint `/chat` devolve audio `.wav` reproduzivel.
- [ ] WER medido em 10 amostras; comparacao sem/com vocabulario documentada.
- [ ] `docs/v01_voz_evidencia.md` preenchido com a evidencia.
- [ ] Commit `v0.1 - Voz` feito.

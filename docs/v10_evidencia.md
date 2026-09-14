# v1.0 — Evidência medida (produção)

> Todos os números abaixo foram **medidos intra-sessão** em 2026-09-14, contra o
> Space em produção (`rafaelang/assistente-master-iag`). Saídas de comando são
> dado intermediário; a evidência oficial é este documento.

## 1. URL pública funcionando — saúde do container

| Verificação | Resultado |
|---|---|
| `GET /` (frontend) | **HTTP 200** em ~0,86 s |
| `GET /saude` (estado) | **HTTP 200** em ~0,60 s |
| chunks indexados | **3.517** |
| corpus_v3 (texto+imagem) | **294 chunks · 22 imagens** |
| LLM configurado | `true` (DeepSeek) |

```json
{
  "rag": { "chunks_indexados": 3517, "corpus_v3": { "ativo": true, "chunks_total": 294, "chunks_imagem": 22 } },
  "llm": { "provedor": "deepseek", "configurado": true }
}
```

O `GET /saude` prova que o **container subiu e o corpus foi baixado do dataset
privado no boot** (o corpus não está no repo público — `snapshot_download` com
`HF_TOKEN_READ`).

### 1.1 Smoke test fim-a-fim no Space (pergunta real)

`POST /chat` com `texto="O que é uma GAN?"` (2026-09-14, após a API estabilizar):

| Campo | Valor |
|---|---|
| HTTP | **200 em 14,7 s** (latência LLM 14,2 s) |
| rota / agente | simples → `gerador-flash` → **fallback `gerador-pro`** |
| motivo do fallback | `gerador-flash: resposta vazia` |
| abstenção | `false` |
| resposta | "Uma GAN ... arquitetura baseada em competição entre dois modelos, o gerador e o discriminador ... [1][2][3][4]" |
| referências | `dgl_aula01_gan_vae.pdf` — páginas 7, 11, 23, 20 |

O fluxo **responde citando a fonte** e o **fallback cruzado resgata** o retorno
vazio do `flash` (encaminhando para o `pro`), sem travar nem alucinar — é o
comportamento de degradação graciosa projetado na v0.4 e coberto pelo §4.

## 2. Diagnóstico do endpoint de modelos (falha #1 — nomes)

`GET https://api.deepseek.com/models` (2026-09-13/14, intra-sessão):

```json
{"object":"list","data":[
  {"id":"deepseek-flash","object":"model","owned_by":"deepseek"},
  {"id":"deepseek-v4-pro","object":"model","owned_by":"deepseek"}]}
```

O config usava `deepseek-chat` e `deepseek-v4-flash` — **nomes inexistentes** no
endpoint. Chamadas a nomes inválidos **penduram** (não retornam 400). Medido:

| Modelo (request) | max_tokens | Resultado |
|---|---|---|
| `deepseek-chat` (não existe) | 50 | **timeout** (25–60 s, sem resposta) |
| `deepseek-v4-flash` (não existe) | 100 | **timeout** (sem resposta) |
| `deepseek-v4-pro` ✓ | 1500 | **HTTP 200 em 1,7 s**, `content='SIMPLES'`, `finish=stop` |
| `deepseek-flash` ✓ (nome correto) | 1500 | **timeout** (instável no momento — ver §3) |

**Correção aplicada:** `AGENTE_MODELO_FLASH` e `DEEPSEEK_MODEL` → `deepseek-flash`;
`AGENTE_MODELO_PRO`/`JUIZ_MODEL` já estavam corretos (`deepseek-v4-pro`).

## 3. Falha #2 — boot do container (SIGPIPE, exit 141)

Histórico de estágios do runtime medido via `get_space_runtime`:

| sha | estágio | causa |
|---|---|---|
| `42a5186` (1º deploy) | RUNNING | ok |
| `b28f4a9` (2º deploy) | RUNTIME_ERROR | exit 141 no `snapshot_download` (tqdm + pipe fechado) — `Fetching 240 files: 66%` |
| `494e9a7` (3º deploy) | RUNTIME_ERROR | exit 141 **após** download: `ls /app/data/raw \| head -5` com `pipefail` |
| `536f737` (4º deploy) | **RUNNING** | fix: `disable_progress_bars` + retry + `ls \| wc -l` |
| `0acf45c` (5º deploy) | **RUNNING** | fix adicional: timeout no cliente LLM |

O estágio final é **RUNNING** com o corpus carregado (§1).

## 4. Falha #3 — degradação graciosa (timeout no cliente)

Antes: `OpenAI()` sem timeout → chamada a API fora do ar pendurava ~10 min.
Depois: `timeout=90` + `max_retries=1` (`LLM_TIMEOUT_S`/`LLM_MAX_RETRIES`).

Fluxo de fallback verificado no código (`agentes.py:257`): quando todas as
rotas falham, o assistente retorna `NAO_SEI` + `motivo="falha em ambas as
rotas"` (abstenção explícita, não travamento).

## 5. Custo por mil consultas

| Rota | US$/pergunta | **US$/1.000 consultas** |
|---|---|---|
| cascade (default) | 0,00027 | **0,27** |
| frontier `pro` forçado | 0,0015 | **1,50** |

Fonte dos coeficientes: `scripts/avaliadores/avaliar_cascade_v05b.py`
(`PRECO_IN=0.27`, `PRECO_OUT=1.10` por M tokens) × contagem real de tokens do
fim-a-fim v0.6 (`docs/v06_evidencia.md` §2).

## 6. Testes

```
$ pytest tests/ -q --deselect tests/test_v02.py::test_rag_responder_pipeline \
    --deselect tests/test_v02.py::test_abstencao_negativa \
    --deselect tests/test_v03.py::test_descrever_imagem_integracao
83 passed, 3 deselected in 201.21s
```

- Os **3 deselected** são integrações que chamam a **API remota** (fora do ar
  no fechamento) — não afetam a suíte de unidade. Total da suíte: **86 testes**.
- `tests/test_v04.py` (27 testes, inclui o fluxo multiagente com
  `AGENTE_MODELO_FLASH`) passou **verde** após a correção dos nomes de modelo.

## 7. Reprodução

```bash
cd projeto_final && source .venv/bin/activate
pytest tests/ -q                          # suíte (exclui os 3 e2e de API se a API estiver fora)
bash deploy/deploy.sh --dry-run           # monta o bundle sem publicar
bash deploy/deploy.sh                     # publica Space + dataset privado

# saúde do Space em produção
curl -s https://rafaelang-assistente-master-iag.hf.space/saude | python3 -m json.tool
curl -s -X POST https://rafaelang-assistente-master-iag.hf.space/chat -F "texto=O que é uma GAN?"
```

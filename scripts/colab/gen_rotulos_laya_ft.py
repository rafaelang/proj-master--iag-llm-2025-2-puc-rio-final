"""Gera o dataset de TREINO do fine-tune do Laya como roteador (cenario v0.7).

Cenario: `docs/v06_roteador_laya_ft.md` — fine-tune de `laya-multilingual`
(RLCD, `research/scripts/finetune_single_device.py`) para SIMPLES/COMPLEXA.

Receita (mesmo espirito do P5/P7 do golden, aplicada FORA do golden):
  1. geracao com o pro ancorada em chunks reais (perguntas de ALUNO no dominio
     do curso, dirigidas ao rotulo-alvo simples|complexa);
  2. auditoria INDEPENDENTE com o pro: o auditor ve APENAS a pergunta e aplica
     o criterio congelado (`data/golden_set/adaptacao/criterio_rotulo.md`),
     sem saber o rotulo-alvo — so entra se auditor == alvo;
  3. dedup normalizada + jaccard contra TUDO que e regua de avaliacao
     (golden v0.7 516Q, golden v0.2 20Q, artefato de roteamento 34 itens);
  4. split fixo 500 treino / 100 teste (held-out, estratificado), com
     augmentacao de typo so no treino (mesmos operadores do
     `gerar_dataset_roteador.py`, simulando ASR pt-BR).

Saida (dado intermediario, fora do Git) em data/processed/rag_v4/:
  roteador_treino_laya.jsonl    treino no formato do upstream
                                {"state","questions","gold"} (choice + noul)
  roteador_treino_laya_teste.json  held-out no formato do
                                `avaliar_roteador_laya.py` (id/tipo/texto/rotulo)
  roteador_treino_laya_meta.json   auditoria/estatisticas do gerador

NAO toca o golden (somente leitura, para dedup) e nao e golden set — e dataset
de treino de um cenario de hipotese, como o artefato de 34 itens.

Roda localmente (API DeepSeek pro via gateway):
  python scripts/colab/gen_rotulos_laya_ft.py --alvo 600
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "avaliadores"))

from projeto_final import config
from projeto_final.rag.pipeline import carregar_chunks

from avaliar_roteador_laya import CRITERIOS, INSTRUCAO, INSTRUCAO_NOUL

load_dotenv()
API_KEY = config.DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise SystemExit("DEEPSEEK_API_KEY nao configurada")
cliente = OpenAI(api_key=API_KEY, base_url=config.DEEPSEEK_BASE_URL,
                 default_headers={"x-opencode-session": os.getenv("OPENCODE_SESSION_ID", "projeto-final-assistente"),
                                  "User-Agent": "projeto-final-assistente/1.0"},
                 timeout=120, max_retries=2)

GEN_MODELO = os.environ.get("GEN_MODELO", config.AGENTE_MODELO_PRO)
SEED = int(os.environ.get("SEED", "42"))
MIN_CHUNK = 260
SAIDA_DIR = config.PROCESSED_DIR / "rag_v4"
SAIDA_TREINO = SAIDA_DIR / "roteador_treino_laya.jsonl"
SAIDA_TESTE = SAIDA_DIR / "roteador_treino_laya_teste.json"
SAIDA_META = SAIDA_DIR / "roteador_treino_laya_meta.json"

# criterio congelado (rotulo do roteador) — copia textual do criterio do
# artefato de roteamento; o auditor nao ve rotulo-alvo, so isto.
CRITERIO_CONGELADO = (
    "SIMPLES = pergunta factual e direta, de resposta unica, que pode ser "
    "respondida com uma unica fonte.\n"
    "COMPLEXA = pergunta composta (duas ou mais partes), comparativa, ambigua "
    "ou que exige raciocinio sobre multiplas fontes."
)


def _limpar_ansi(txt: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", txt)


def _chat_bigpickle(system: str, user: str, timeout: int = 180) -> str:
    """Fallback: grande-pro sao gerados via `opencode run` com o modelo
    big-pickle (sai do gateway zen que pode ficar esvaziando)."""
    d = Path(os.environ.get("BIGP_DIR", "/tmp/opencode/bigp")) / uuid.uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    modelo = os.environ.get("BIGP_MODELO", "opencode/big-pickle")
    cmd = ["opencode", "run", "--pure", "--dir", str(d), "-m", modelo,
           f"{system}\n\n{user}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return _limpar_ansi(r.stdout).strip()
    except Exception as e:
        print(f"  [bigpickle] erro: {e}", flush=True)
        return ""
    finally:
        for p in d.iterdir():
            try:
                if p.is_dir():
                    _remove_tree(p)
                else:
                    p.unlink()
            except OSError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass


def _remove_tree(p: Path) -> None:
    for q in p.iterdir():
        if q.is_dir():
            _remove_tree(q)
        else:
            q.unlink()
    p.rmdir()


def _chat(modelo: str, system: str, user: str, max_tokens: int = 600,
          temperature: float = 0.3, tentativas: int = 3) -> str:
    for t in range(tentativas):
        try:
            r = cliente.chat.completions.create(
                model=modelo,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature,
            )
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt
            print(f"  [api] vazio ({t + 1}/{tentativas})", flush=True)
        except Exception as e:
            print(f"  [api] erro ({t + 1}/{tentativas}): {e}", flush=True)
        time.sleep(2 + 2 * t)
    # gateway zen esvaziando -> big-pickle (instrucao do usuario)
    print("  [bigpickle] fallback acionado", flush=True)
    return _chat_bigpickle(system, user)


def _extrair_json(txt: str) -> dict:
    if not txt:
        return {}
    t = re.sub(r"```[a-z]*", "", txt)
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalizar(q: str) -> str:
    return re.sub(r"[^a-z0-9]", "", q.lower())


def _tokens(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", q.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sem_acento(s: str) -> str:
    t = unicodedata.normalize("NFKD", s)
    return "".join(c for c in t if not unicodedata.combining(c))


def _para_pra(s: str) -> str:
    return _sem_acento(s).replace("para que serve", "pra que serve")


def _drop_meio(s: str) -> str:
    t = _sem_acento(s)
    i = len(t) // 2
    return t[:i] + t[i + 1:]


def _troca_meio(s: str) -> str:
    t = _sem_acento(s)
    i = len(t) // 2
    if i + 1 >= len(t):
        return _drop_meio(s)
    return t[:i] + t[i + 1] + t[i] + t[i + 2:]


TYPO_OPS = [_sem_acento, _para_pra, _drop_meio, _troca_meio]

GEN_SYS = (
    "Voce gera PERGUNTAS DE ALUNO para um assistente sobre o curso Master IAG "
    "& LLM (PUC-Rio). Responda APENAS com JSON: {\"pergunta\": \"...\"}"
)

AUDIT_SYS = (
    "Voce classifica perguntas de um assistente do curso Master IAG & LLM "
    "(PUC-Rio) em dois tipos de rota:\n\n"
    f"{CRITERIO_CONGELADO}\n\n"
    "Responda APENAS com JSON: "
    "{\"rotulo\": \"SIMPLES\" ou \"COMPLEXA\", \"motivo\": \"<curto>\"}"
)


def gerar_pergunta(alvo: str, chunks: list[dict]) -> str:
    if alvo == "SIMPLES":
        c = chunks[0]
        user = (
            f"Trecho (fonte: {c['doc_id']}):\n\n\"{c['texto'][:1500]}\"\n\n"
            "Gere UMA pergunta SIMPLES: factual, direta, com resposta unica, "
            "que possa ser respondida com ESTE unico trecho. Frase de aluno, "
            "curta (ate 160 caracteres), terminando em '?'."
        )
    else:
        a, b = chunks
        user = (
            f"Trecho 1 (fonte: {a['doc_id']}):\n\n\"{a['texto'][:1200]}\"\n\n"
            f"Trecho 2 (fonte: {b['doc_id']}):\n\n\"{b['texto'][:1200]}\"\n\n"
            "Gere UMA pergunta COMPLEXA: composta (duas ou mais partes), "
            "comparativa, ambigua ou que exige raciocinio sobre as fontes acima. "
            "Frases de aluno, no maximo 2 frases (ate 230 caracteres), "
            "terminando em '?'."
        )
    for tentativa in range(2):
        obj = _extrair_json(_chat(GEN_MODELO, GEN_SYS, user, max_tokens=400,
                                  temperature=0.4))
        p = (obj.get("pergunta") or "").strip()
        if _valida(p):
            return p
        user = user + (
            f"\n\nSua ultima tentativa foi: \"{p[:400]}\" — reescreva em no "
            "maximo 2 frases (ate 200 caracteres), terminando em '?'."
        )
    return p


def auditar(pergunta: str) -> tuple[str, str]:
    obj = _extrair_json(_chat(GEN_MODELO, AUDIT_SYS, pergunta,
                              max_tokens=200, temperature=0.0))
    rot = str(obj.get("rotulo", "")).strip().upper()
    rot = rot if rot in ("SIMPLES", "COMPLEXA") else ""
    return rot, str(obj.get("motivo", ""))[:80]


def _regua_avaliacao() -> list[str]:
    """Textos que NAO podem aparecer no treino (regua de avaliacao)."""
    textos: list[str] = []
    for nome in ("perguntas_v07.json", "perguntas_v06.json",
                 "perguntas_v05b.json", "perguntas.json"):
        p = config.GOLDEN_SET_DIR / "rag" / nome
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            textos += [q["pergunta"] for q in d["perguntas"]]
    art = SAIDA_DIR / "roteador_rotulos.json"
    if art.exists():
        d = json.loads(art.read_text(encoding="utf-8"))
        textos += [i["texto"] for i in d["itens"]]
    return textos


def _valida(pergunta: str) -> bool:
    return 12 <= len(pergunta) <= 240 and "?" in pergunta


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset de treino do fine-tune do Laya (roteador)")
    ap.add_argument("--alvo", type=int, default=600, help="sobreviventes apos auditoria+dedup")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--teste", type=int, default=100, help="tamanho do held-out")
    ap.add_argument("--aug-typo", type=float, default=0.20,
                    help="fracao do treino com variantes de typo (ASR)")
    ap.add_argument("--sobre", type=float, default=1.8,
                    help="oversample de candidatos por sobrevivente")
    ap.add_argument("--novo", action="store_true",
                    help="ignora candidatos ja persistidos (recomeca do zero)")
    a = ap.parse_args()

    random.seed(SEED)
    chunks = [c for c in carregar_chunks() if len(c.get("texto", "")) >= MIN_CHUNK]
    random.shuffle(chunks)
    por_doc: dict[str, list[dict]] = {}
    for c in chunks:
        por_doc.setdefault(c["doc_id"], []).append(c)
    docs = list(por_doc)
    random.shuffle(docs)

    # pares (alvo, chunks) — metade simples (1 chunk), metade complexa (2 docs).
    # oversample >1: auditoria/dedup descartam parte dos candidatos.
    n_tarefas = int(a.alvo * a.sobre)
    tarefas: list[tuple[str, list[dict]]] = []
    i_doc = 0
    while len(tarefas) < n_tarefas:
        for alvo in ("SIMPLES", "COMPLEXA"):
            if len(tarefas) >= n_tarefas:
                break
            if alvo == "SIMPLES":
                c = chunks[len(tarefas) % len(chunks)]
                tarefas.append((alvo, [c]))
            else:
                d1 = docs[i_doc % len(docs)]
                d2 = docs[(i_doc + 1) % len(docs)]
                i_doc += 1
                tarefas.append((alvo, [por_doc[d1][0], por_doc[d2][0]]))
    random.shuffle(tarefas)

    regua = [normalizar(t) for t in _regua_avaliacao()]
    regua_toks = [_tokens(t) for t in _regua_avaliacao()]

    def trabalhar(args: tuple[int, tuple[str, list[dict]]]) -> dict | None:
        idx, (alvo, chs) = args
        pergunta = gerar_pergunta(alvo, chs)
        if not _valida(pergunta):
            print(f"  [invalida:{len(pergunta)}] {pergunta[:60]!r}", flush=True)
            return None
        k = normalizar(pergunta)
        if k in regua:
            print(f"  [dedup-exato] {pergunta[:60]}", flush=True)
            return None
        tk = _tokens(pergunta)
        if any(jaccard(tk, rt) > 0.80 for rt in regua_toks):
            print(f"  [dedup-fuzzy] {pergunta[:60]}", flush=True)
            return None
        rot, motivo = auditar(pergunta)
        if not rot:
            print(f"  [auditor-vazio] {pergunta[:60]}", flush=True)
            return None
        if rot != alvo:
            print(f"  [discordei] alvo={alvo} auditor={rot}: {pergunta[:60]} "
                  f"({motivo})", flush=True)
            return {"ok": False, "alvo": alvo, "auditor": rot}
        if idx % 25 == 0:
            print(f"  [ok {idx}] {alvo}: {pergunta[:70]}", flush=True)
        return {"ok": True, "pergunta": pergunta, "rotulo": rot,
                "auditor": rot, "motivo": motivo,
                "docs": [c["doc_id"] for c in chs]}

    print(f"Gerando: alvo={a.alvo} · workers={a.workers} · sobre={a.sobre} · "
          f"modelo={GEN_MODELO}", flush=True)

    # resume: aceitos persistidos em lote incremental (sobrevive a queda/reinicio).
    CAND = SAIDA_DIR / "roteador_treino_laya_candidatos.jsonl"
    unicos: list[dict] = []
    vistos: set[str] = set()
    previos = 0
    if not a.novo and CAND.exists():
        for linha in CAND.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if not linha:
                continue
            obj = json.loads(linha)
            k = normalizar(obj["pergunta"])
            if obj.get("ok") and k not in vistos:
                vistos.add(k)
                unicos.append(obj)
        print(f"[resume] {len(unicos)} sobreviventes previos em {CAND.name}",
              flush=True)
    previos = len(unicos)
    CAND.parent.mkdir(parents=True, exist_ok=True)

    def _persiste(obj: dict) -> None:
        with CAND.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    disc: list[dict] = []
    n_nulos = 0
    rodada = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(trabalhar, (i, t)) for i, t in enumerate(tarefas)]
        for fu in as_completed(futs):
            if len(unicos) >= a.alvo:
                fu.cancel()
                continue
            rodada += 1
            try:
                r = fu.result(timeout=240)
            except FutureTimeout:
                print("  [timeout] tarefa abandonada (worker segue ocupado)",
                      flush=True)
                n_nulos += 1
                continue
            except Exception as e:
                print(f"  [excecao] {e}", flush=True)
                n_nulos += 1
                continue
            if r is None:
                n_nulos += 1
                continue
            if r.get("ok") is False:
                disc.append(r)
                continue
            k = normalizar(r["pergunta"])
            if k in vistos:
                continue
            vistos.add(k)
            unicos.append(r)
            _persiste(r)
            if len(unicos) % 50 == 0:
                print(f"  {{unicos={len(unicos)}}}", flush=True)

    # shuffle fixo + split (teste ate --teste, resto treino; robusto a sobra)
    random.seed(SEED)
    random.shuffle(unicos)
    n_teste = min(a.teste, max(0, len(unicos) - 50))
    teste, treino = unicos[:n_teste], unicos[n_teste:]

    # --- treino: JSONL do upstream (choice + noul por estado) ---
    linhas = []
    n_aug = 0
    for r in treino:
        estados = [r["pergunta"]]
        if random.random() < a.aug_typo:
            op = random.choice(TYPO_OPS)
            var = op(r["pergunta"])
            if var != r["pergunta"]:
                estados.append(var)
                n_aug += 1
        for estado in estados:
            complexa = r["rotulo"] == "COMPLEXA"
            linhas.append({
                "state": estado,
                "questions": {
                    "rota": {"type": "choice", "instructions": INSTRUCAO,
                             "criteria": CRITERIOS},
                    "complexa": {"type": "noul", "instructions": INSTRUCAO_NOUL},
                },
                "gold": {
                    "rota": {"probabilities": {
                        "simples": 0.0 if complexa else 1.0,
                        "complexa": 1.0 if complexa else 0.0}},
                    "complexa": {"probabilities": {
                        "false": 0.0 if complexa else 1.0,
                        "true": 1.0 if complexa else 0.0}},
                },
            })

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    with SAIDA_TREINO.open("w", encoding="utf-8") as f:
        for ln in linhas:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")

    # --- teste: formato do avaliar_roteador_laya.py ---
    n_s = sum(1 for r in teste if r["rotulo"] == "SIMPLES")
    dados_teste = {
        "meta": {
            "descricao": "held-out do dataset de treino do fine-tune Laya — NAO e golden",
            "criterio": "SIMPLES=factual/direta/1 fonte · COMPLEXA=composta/comparativa/ambigua/multifonte",
            "fonte": str(SAIDA_TREINO),
            "n_canonicas": len(teste),
            "n_typos": 0,
        },
        "resumo_rotulos": {"SIMPLES": n_s, "COMPLEXA": len(teste) - n_s},
        "itens": [
            {"id": i + 1, "tipo": "canonica", "texto": r["pergunta"],
             "rotulo": r["rotulo"]}
            for i, r in enumerate(teste)
        ],
    }
    SAIDA_TESTE.write_text(json.dumps(dados_teste, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    meta = {
        "descricao": "geracao do dataset de treino do fine-tune do Laya (roteador)",
        "modelo_gerador": GEN_MODELO,
        "alvo": a.alvo,
        "seed": SEED,
        "tentativas_rodada": rodada,
        "resumidos_previos": previos if not a.novo else 0,
        "candidatos_invalidos_ou_vazios": n_nulos,
        "aprovados_apos_auditoria": len(unicos) + len(disc),
        "discordancias_auditoria": len(disc),
        "taxa_concordancia": round(len(unicos) / max(1, len(unicos) + len(disc)), 3),
        "un": len(unicos),
        "split": {"treino": len(treino), "teste": len(teste)},
        "estados_treino_com_typo": n_aug,
        "linhas_jsonl": len(linhas),
        "saidas": {"treino": str(SAIDA_TREINO), "teste": str(SAIDA_TESTE)},
    }
    SAIDA_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    print("\n=== RESUMO ===")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"discordancias da auditoria: {len(disc)} · invalidos/vazios: {n_nulos}")


if __name__ == "__main__":
    main()

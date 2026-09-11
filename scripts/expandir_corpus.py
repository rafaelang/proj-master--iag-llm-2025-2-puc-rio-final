"""Expande o corpus didatico de data/raw a partir de AULAS/ (v0.5b).

Varre as pastas ``AULAS/{TDP,DPIA,NLP,PAI,DGL,OFICINA-TDP,PROJ}``, deduplica por
sha1 e copia para ``data/raw`` usando nomes normalizados que seguem a convencao
``<disciplina>_aula<NN>_<tema>[__sufixo].<ext>``.

Regras:
- Os arquivos ja presentes em ``data/raw`` sao PRESERVADOS (mesmo nome) - o
  golden set da v0.5 referencia esses nomes em ``docs_esperados``.
- Arquivos novos recebem nome deterministico a partir do titulo da aula
  (pasta "Aula NN - <tema>") e/ou do nome do proprio arquivo.
- Duplicatas de conteudo (sha1) entre pastas/disciplinas sao copiadas uma unica
  vez; as demais ocorrencias ficam registradas como ``duplicatas`` no manifesto.
- Ruido e excluido: venvs, node_modules, caches, snapshot ``PROJ/projeto``,
  pastas ``antiga/antigo`` e arquivos de referencia soltos.

O manifesto (``data/golden_set/corpus_manifest.json``) entra no Git e documenta,
para cada arquivo do corpus: origem em AULAS, disciplina, aula, sha1, tamanho,
se e pre-existente a v0.5 ou novo na v0.5b, e as duplicatas descartadas.

Uso:
    python scripts/expandir_corpus.py --dry-run      # apenas relatorio
    python scripts/expandir_corpus.py                 # copia + escreve manifesto
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from loguru import logger

AULAS_ROOT = Path(
    "/home/rafael/projetos/hermes_study/POS_GenAI/AULAS"
)
DISCIPLINAS = ["TDP", "DPIA", "NLP", "PAI", "DGL", "OFICINA-TDP", "PROJ"]

RAIZ = Path(__file__).resolve().parent.parent
RAW_DIR = RAIZ / "data" / "raw"
MANIFEST = RAIZ / "data" / "golden_set" / "corpus_manifest.json"

# Extensoes que entram no corpus do projeto_final (v0.5b: pdf/md/ipynb/pptx).
EXTENSOES = {".pdf", ".md", ".markdown", ".ipynb", ".pptx"}
# Documentos de referencia soltos na raiz do AULAS (nao sao aulas).
EXCLUIR_RAIZ = {
    "AGENTS.md",
    "docs",
    "Cronograma.xlsx",
    "covid.pdf", "covid.txt",
    "dialnet.pdf", "dialnet.txt",
    "doc.html", "ele.html", "gp.html",
    "ppgeet_docentes.html", "ppgeet_doc.html",
    "forero_cv.pdf", "forero_cv.txt",
    "maxwell.pdf", "maxwell.txt",
    "prossim.pdf", "prossim.txt",
    "prossim80.pdf", "prossim80.txt",
    "slides_ed25519", "slides_ed25519.pub",
    "uerj.pdf", "uerj.txt",
}
# Nomes de pasta/arquivo que denotam lixo/duplicacao historica.
DIRS_IGNORAR = {
    ".venv", "venv", "node_modules", "site-packages", "__pycache__",
    ".pytest_cache", ".git", ".ipynb_checkpoints", "dist-info", "egg-info",
    "projeto", "antiga", "antigo", "old", "lixo",
}

# regex do "tema" a partir do nome da pasta titulada (ex.: "Aula 01 - Tema",
# "Oficina 04 - Tema", "Aula-10 - Tema", "Aula Extra -Tema")
_RE_TITULO = re.compile(
    r"^\s*(?:aula|oficina)\s*(?:extra)?\s*[0-9]{0,2}\s*[-–—:.]\s*(?P<tema>.+?)\s*$",
    re.IGNORECASE,
)
# prefixos esperados em nome de arquivo (removidos no slug), ex.: "TDP - Aula 05"
_RE_PREFIXO_NOME = re.compile(
    r"^\s*(?:"
    r"nlp\s*[0-9.]*\s*[-–—]?\s*"
    r"|llm[-_]?master[-_\s]*"
    r"|genai[-_\s]*llms[-_\s]*"
    r"|tdp\s*[-–—]\s*"
    r"|oficina\s*[0-9]+\s*[-–—]\s*"
    r"|aula\s*"
    r")+",
    re.IGNORECASE,
)
# sufixos "(1)", "(2)", "_2", " (2) (1)" no fim do nome do arquivo
_RE_SUFIXO_VERSAO = re.compile(r"\s*\([0-9]+\)\s*$|_\d+$")


def _slug(texto: str, maxlen: int = 60) -> str:
    """Normaliza texto para slug: minusculas, sem acento, [a-z0-9_]."""
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = t.strip("_")
    t = re.sub(r"_+", "_", t)
    return t[:maxlen].strip("_")


def _slug_arquivo(stem: str) -> str:
    """Slug do nome do arquivo, removendo prefixos/sufixos repetitivos."""
    s = stem
    s = _RE_SUFIXO_VERSAO.sub("", s)
    s = s.strip()
    s = _RE_PREFIXO_NOME.sub("", s).strip()
    s = re.sub(r"^\s*(c[oó]pia\s*de|copy\s*of|com_copia)\s+", "", s,
               flags=re.IGNORECASE)
    s = _slug(s)
    return s or _slug(stem)


def _tema_da_pasta(nome_pasta: str) -> str | None:
    """Extrai o tema (slug) do nome de uma pasta titulada, se houver."""
    nome_pasta = nome_pasta.strip()
    # pastas internas tipo "exercício 1" nao sao titulos de aula
    for padrao in (
        r"^exerc[ií]c\w*\s*\d*$",
        r"^exemplo[s]?\s*\d*$",
        r"^scripts$",
        r"^embeddings$",
    ):
        if re.search(padrao, nome_pasta, re.IGNORECASE):
            return None
    m = _RE_TITULO.match(nome_pasta)
    if not m:
        return None
    tema = _slug(m.group("tema"))
    if len(tema) < 3:
        return None
    return tema


def _sha1(caminho: Path) -> str:
    h = hashlib.sha1()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def _eh_ignorado(rel: Path) -> bool:
    """True se qualquer parte do caminho denota lixo a excluir."""
    return any(p in DIRS_IGNORAR for p in rel.parts)


def _verificar_copia(src: Path, dst: Path) -> bool:
    return (dst.exists() and dst.stat().st_size == src.stat().st_size
            and _sha1(src) == _sha1(dst))


def _coletar_fontes(aulas: Path) -> tuple[list[tuple[Path, str, str]], list[dict]]:
    """Retorna (arquivos_significativos, excluidos).

    Cada item de arquivos_significativos: (caminho, disciplina, aula).
    ``aula`` pode ser None para decks raiz (slides_*.md) e para o indice.
    """
    fontes: list[tuple[Path, str, str]] = []
    excluidos: list[dict] = []
    # indice geral do curso (origem do curso_indice.md preservado)
    indice = aulas / "INDICE.md"
    if indice.is_file():
        fontes.append((indice, "curso", None))
    for parte in sorted(aulas.iterdir()):
        if parte.name not in DISCIPLINAS or parte.is_file():
            continue
        disciplina = _slug(parte.name)
        # decks raiz da disciplina (slides_*.md) -- fora de pasta aulaNN
        for deck in sorted(parte.iterdir()):
            if deck.is_file() and deck.suffix.lower() in EXTENSOES:
                if deck.name in EXCLUIR_RAIZ or _eh_ignorado(Path(deck.name)):
                    continue
                fontes.append((deck, disciplina, None))
        for caminho in sorted(parte.rglob("*")):
            if not caminho.is_file():
                continue
            rel = caminho.relative_to(aulas)
            if _eh_ignorado(rel):
                continue
            if caminho.suffix.lower() not in EXTENSOES:
                continue
            aula = None
            for p in reversed(rel.parts):
                if re.fullmatch(r"aula\d{2}", p, re.IGNORECASE):
                    aula = p.lower()
                    break
            if aula is None:
                excluidos.append({
                    "caminho": str(rel),
                    "motivo": "fora de pasta aulaNN",
                })
                continue
            fontes.append((caminho, disciplina, aula))
    return fontes, excluidos


def _caminho_titulo_ancestral(caminho: Path, aulas: Path) -> str | None:
    """Slug do tema da pasta titulada mais proxima (ou None)."""
    rel = caminho.relative_to(aulas)
    for parte in reversed(rel.parents):  # mais proximo primeiro
        tema = _tema_da_pasta(parte.name)
        if tema:
            return tema
    return None


def _nome_destino(
    caminho: Path,
    aulas: Path,
    disciplina: str,
    aula: str,
    ext: str,
) -> str | None:
    """Gera um nome deterministico e unico para o novo arquivo.

    Retorna None se o conteudo ja existe no corpus (dedup por sha1) - nesse
    caso o nome ja foi usado e o registro deve apontar para a ocorrencia previa.
    """
    stem = caminho.stem
    tema = _caminho_titulo_ancestral(caminho, aulas)
    if tema and aula:
        base = f"{disciplina}_{aula}_{tema}"
        chave = _slug_arquivo(stem)
        # evita sufixo redundante quando o stem ja e subconjunto do tema
        # (ex.: deck "TDP - Aula 05" em pasta "Aula 05 - SQL para Consulta...")
        if (chave and chave != tema
                and chave not in tema and tema not in chave):
            base = f"{base}_{chave}"
    elif not aula and stem.startswith("slides_"):
        base = f"{disciplina}_{_slug(stem)}"
    elif not aula:
        base = f"{disciplina}_{_slug_arquivo(stem)}"
    else:
        base = f"{disciplina}_{aula}_{_slug_arquivo(stem)}"
    return (base + ext).lower()


def expandir(dry_run: bool = True) -> None:
    if not AULAS_ROOT.is_dir():
        logger.error("AULAS_ROOT nao encontrado: {}", AULAS_ROOT)
        sys.exit(1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1) arquivos ja presentes em data/raw - preservados pelo nome
    existentes: dict[str, str] = {}       # sha -> nome (ja no corpus)
    usados: dict[str, Path] = {}          # nome -> caminho (evita colisao)
    for f in sorted(RAW_DIR.glob("*")):
        if f.is_file():
            existentes[_sha1(f)] = f.name
            usados[f.name] = f
    # nome pre-existente (por sha1) para manter imutavel; caso contrario gera
    # 2) coleta de fontes
    fontes, excluidos = _coletar_fontes(AULAS_ROOT)
    logger.info("{} arquivos significativos coletados ({} excluidos ruido).",
                len(fontes), len(excluidos))

    manifest: list[dict] = []
    novos = dups = mantidos = 0
    # sha -> entry (para registrar duplicatas de conteudo)
    entry_por_sha: dict[str, dict] = {}

    for caminho, disciplina, aula in fontes:
        sha = _sha1(caminho)
        ext = caminho.suffix.lower()
        rel = str(caminho.relative_to(AULAS_ROOT))

        # ja está no corpus (pre-existente v0.5 OU copiado nesta execucao)?
        if sha in entry_por_sha:
            entry_por_sha[sha].setdefault("duplicatas", []).append(rel)
            dups += 1
            continue
        if sha in existentes:
            nome = existentes[sha]
            pre = True
            mantidos += 1
        else:
            nome = _nome_destino(caminho, AULAS_ROOT, disciplina, aula, ext)
            origem = caminho.name
            while nome in usados:
                i = 2
                base = Path(nome).stem
                nome = f"{base}_{i}{ext}"
                i += 1
            pre = False
            novos += 1

        destino = RAW_DIR / nome
        entry = {
            "arquivo": nome,
            "disciplina": disciplina,
            "aula": aula,
            "extensao": ext,
            "sha1": sha,
            "tamanho": caminho.stat().st_size,
            "origem": rel,
            "pre_v05": pre,
            "novo_v05b": not pre,
            "duplicatas": [],
        }
        entry_por_sha[sha] = entry
        usados[nome] = destino
        manifest.append(entry)

        if not dry_run and not destino.exists():
            shutil.copy2(caminho, destino)
            if not _verificar_copia(caminho, destino):
                raise RuntimeError(f"Copia com integridade falhou: {nome}")
            logger.debug("copiado {} -> {}", caminho.name, nome)
        elif destino.exists():
            entry["ja_existia"] = True

    # 3) escrito o manifesto (mesmo em dry-run, para revisao)
    out = {
        "gerado_em": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "origem": str(AULAS_ROOT),
        "formato_incluidos": sorted(EXTENSOES),
        "resumo": {
            "fontes_significativas": len(fontes),
            "mantidos_preservados": mantidos,
            "novos_copiados": novos,
            "duplicatas_descartadas": dups,
            "excluidos_ruido": len(excluidos),
            "total_corpus": len(manifest),
        },
        "excluidos_ruido": sorted(excluidos, key=lambda d: d["caminho"]),
        "excluidos_raiz": sorted(EXCLUIR_RAIZ),
        "corpus": manifest,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    logger.info(
        "Resumo: {} preservados | {} novos | {} duplicatas | corpus total {}",
        mantidos, novos, dups, len(manifest),
    )
    if dry_run:
        logger.warning("Dry-run: nada foi copiado. Manifesto de revisao em {}",
                       MANIFEST)
    else:
        total = sum(1 for _ in RAW_DIR.glob("*") if _.is_file())
        logger.info("Corpus expandido em {} ({} arquivos).", RAW_DIR, total)


if __name__ == "__main__":
    expandir(dry_run="--dry-run" in sys.argv)
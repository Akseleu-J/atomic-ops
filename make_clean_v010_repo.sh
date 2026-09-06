#!/usr/bin/env bash
# =============================================================================
# make_clean_v010_repo.sh  (PATCHED)
#
# 1) Создаёт НОВЫЙ локальный репозиторий из текущего исходника (снапшот)
#    как первую публичную версию: 0.1.0 (без истории фикса Kernel B4 из
#    оригинального CHANGELOG -- в новом репо этой истории просто нет).
# 2) Ребрендинг: gdn2-pallas -> atomic_ops
#      - pyproject.toml: name = "atomic_ops"  (pip install atomic_ops)
#      - README.md / CONTRIBUTING.md: pip install команда, git clone URL,
#        cd <repo>, заголовок проекта, project-layout дерево,
#        GitHub-ссылки, badge-ссылки, bibtex citation
#      - atomic_ops/__init__.py: importlib.metadata.version("gdn2-pallas")
#        -> version("atomic_ops")  (иначе __version__ всегда будет 0.0.0.dev0
#        после переименования дистрибутива)
#    Название алгоритма GDN-2 / Gated DeltaNet-2 НЕ трогается -- это не
#    имя пакета, а название архитектуры.
# 3) Roadmap.md -> переименовывается в ROADMAP.md (капс), чтобы совпадать
#    со ссылками [ROADMAP.md](ROADMAP.md) в README.md / KNOWN_LIMITATIONS.md
#    (на GitHub файловая система регистрозависима -- иначе битая ссылка).
#    Содержимое файла НЕ трогается, только имя.
# 4) CHANGELOG.md переписывается заново под 0.1.0, но теперь честно
#    отражает и то, что реально есть в дереве на момент снапшота
#    (LICENSE, beta/gdn2_hybrid.py, ROADMAP.md, KNOWN_LIMITATIONS.md),
#    без утверждений про версии (v0.1.1/v0.1.5), которых в новом чистом
#    репо не существует.
# 5) git init + первый коммит.
# 6) (опционально) создание репозитория на GitHub через `gh` и push,
#    если задана переменная CREATE_REMOTE=1 и установлен GitHub CLI (`gh`).
#
# Использование:
#   chmod +x make_clean_v010_repo.sh
#   ./make_clean_v010_repo.sh <src_dir> <new_dir> [github_user] [github_repo_name]
#
# Примеры:
#   ./make_clean_v010_repo.sh ./gdn2-pallas ./atomic_ops
#   ./make_clean_v010_repo.sh ./gdn2-pallas ./atomic_ops Akseleu-J atomic_ops
#
# Чтобы сразу создать репозиторий на GitHub и запушить (нужен `gh auth login`):
#   CREATE_REMOTE=1 ./make_clean_v010_repo.sh ./gdn2-pallas ./atomic_ops Akseleu-J atomic_ops
# =============================================================================

set -euo pipefail

SRC_DIR="${1:-./gdn2-pallas}"
NEW_DIR="${2:-./atomic_ops}"
GITHUB_USER="${3:-Akseleu-J}"
NEW_REPO_NAME="${4:-atomic_ops}"

OLD_DIST_NAME="gdn2-pallas"      # старое имя дистрибутива (pip install / PyPI)
NEW_DIST_NAME="atomic_ops"       # новое имя дистрибутива (pip install / PyPI)
OLD_REPO_PATH="gdn2-pallas"      # старый путь в GitHub URL (github.com/USER/<этот путь>)
NEW_REPO_PATH="$NEW_REPO_NAME"   # новый путь в GitHub URL

CREATE_REMOTE="${CREATE_REMOTE:-0}"   # 1 = создать репо на GitHub и запушить (требует `gh`)

if [ ! -d "$SRC_DIR" ]; then
    echo "[ERROR] Исходная директория не найдена: $SRC_DIR"
    echo "        Запуск: $0 <src_dir> <new_dir> [github_user] [github_repo_name]"
    exit 1
fi

if [ -e "$NEW_DIR" ]; then
    echo "[ERROR] Целевая директория уже существует: $NEW_DIR"
    echo "        Удали её или укажи другой путь."
    exit 1
fi

echo "=== 1/8: Копирование $SRC_DIR -> $NEW_DIR ==="
mkdir -p "$NEW_DIR"
rsync -a \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='.jax_cache/' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='*.egg-info/' \
    "$SRC_DIR"/ "$NEW_DIR"/

cd "$NEW_DIR"

echo "=== 2/8: pyproject.toml -> version 0.1.0, name -> ${NEW_DIST_NAME} ==="
if [ -f "pyproject.toml" ]; then
    sed -i.bak -E 's/^version = "[0-9]+\.[0-9]+\.[0-9]+"/version = "0.1.0"/' pyproject.toml
    sed -i.bak -E "s/^name = \"${OLD_DIST_NAME}\"/name = \"${NEW_DIST_NAME}\"/" pyproject.toml
    sed -i.bak -E "s#github\.com/${GITHUB_USER}/${OLD_REPO_PATH}#github.com/${GITHUB_USER}/${NEW_REPO_PATH}#g" pyproject.toml
    rm -f pyproject.toml.bak
else
    echo "[WARN] pyproject.toml не найден, пропускаю."
fi

echo "=== 3/8: atomic_ops/__init__.py -> importlib.metadata.version(\"${NEW_DIST_NAME}\") ==="
if [ -f "atomic_ops/__init__.py" ]; then
    sed -i.bak -E "s/_version\(\"${OLD_DIST_NAME}\"\)/_version(\"${NEW_DIST_NAME}\")/" atomic_ops/__init__.py
    rm -f atomic_ops/__init__.py.bak
else
    echo "[WARN] atomic_ops/__init__.py не найден, пропускаю."
fi

echo "=== 4/8: Roadmap.md -> ROADMAP.md (регистр, чтобы совпасть со ссылками) ==="
if [ -f "Roadmap.md" ] && [ ! -f "ROADMAP.md" ]; then
    git_mv_available=0
    mv "Roadmap.md" "ROADMAP.md"
    echo "    переименовано: Roadmap.md -> ROADMAP.md"
elif [ -f "ROADMAP.md" ]; then
    echo "    ROADMAP.md уже существует в нужном регистре, пропускаю переименование."
else
    echo "[WARN] Roadmap.md не найден, пропускаю."
fi
if [ -f "ROADMAP.md" ]; then
    sed -i.bak -E "s#github\.com/${GITHUB_USER}/${OLD_REPO_PATH}#github.com/${GITHUB_USER}/${NEW_REPO_PATH}#g" ROADMAP.md
    rm -f ROADMAP.md.bak
fi

echo "=== 5/8: Перезапись CHANGELOG.md (единый чистый релиз 0.1.0, ребренд, честный список содержимого) ==="
cat > CHANGELOG.md << EOF
# Changelog

All notable user-facing changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-05

First public release under the \`${NEW_DIST_NAME}\` name (renamed from
\`${OLD_DIST_NAME}\` prior to release; no prior published history).

### Added

- Fused GDN-2 forward kernels for TPU v5e in JAX/Pallas:
  Kernel A (chunk scores), Kernel B (WY block solve), Kernel C (recompute),
  Kernel D (inter-chunk scan).
- Fused backward kernels B1-B5 with a \`jax.custom_vjp\` trainable wrapper
  (\`gdn2_pallas_forward_trainable\`) that reuses forward residuals instead of
  recomputing them.
- Automatic fallback: CPU/GPU or \`d_head != 128\` dispatches to a checkpointed
  pure-JAX chunked-WY reference with identical \`wy_eps\` damping semantics.
- \`KernelConfig\` with TPU v5e presets \`KAGGLE_SMALL\` / \`KAGGLE_MEDIUM\` /
  \`KAGGLE_LARGE\`, plus \`estimate_memory\` / \`get_recommended_config\` helpers.
- Numerical-safety sanitizers at every kernel boundary and a per-stage
  diagnostic mode via the \`GDN2_FWD_DIAG=1\` environment flag.
- Layered correctness suite: CPU smoke tests, multi-seed TPU sweeps,
  finite-difference gradient checks, isolated B3-B5 backward-stage tests,
  BF16 dtype-contract checks, \`wy_eps\` damping coverage
  (see \`docs/TESTING_STRATEGY.md\`).
- Speed and memory benchmarks with raw results (\`benchmarks/\`).
- \`beta/gdn2_hybrid.py\`: experimental, opt-in hybrid JAX-forward +
  fused-Pallas-backward path. Not wired into the public dispatcher
  (\`gdn2_forward_trainable\`); see \`KNOWN_LIMITATIONS.md\` section 5 and
  \`ROADMAP.md\` for status and open validation items before this is
  recommended for production training.
- \`ROADMAP.md\`: tracked hypotheses and open performance items (MXU-factorized
  pairwise decay / \`use_centering\`, Kernel B block-solve investigation),
  explicitly marked as not-yet-implemented where applicable.
- \`KNOWN_LIMITATIONS.md\`: documented gaps (forward slower than pure-JAX WY
  forward, VPU-bound pairwise decay, TPU-only / \`d_head=128\` requirement,
  shape constraints, hybrid-path open items, kernel-gap diagnostic).
- MIT \`LICENSE\`.

### Known limitations

- The fused forward is currently slower than the pure-JAX WY forward
  (~0.62x on TPU v5e-8, forward-only). Training steps are backward-dominated,
  so the full cycle is still faster than the best pure-JAX baseline end-to-end.
- The pairwise decay computation (Kernel A / B4) is VPU-bound rather than
  MXU-bound; an MXU-factorized \`use_centering\` alternative is a documented
  hypothesis in \`ROADMAP.md\`, not implemented in this release.
- See \`KNOWN_LIMITATIONS.md\` for the full list, including the experimental
  hybrid path's open validation items.

### Performance (TPU v5e-8, train shape B=8, L=4096, 6 heads, d_head=128)

| Metric | FP32 | BF16 |
| --- | --- | --- |
| fwd+bwd vs associative_scan | 27.2x | 13.3x |
| fwd+bwd vs pure-JAX chunked WY | 2.6x | 3.4x |
| Best-case vs associative_scan (all shapes) | 38.8x | 18.7x |

[Unreleased]: https://github.com/${GITHUB_USER}/${NEW_REPO_PATH}/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/${GITHUB_USER}/${NEW_REPO_PATH}/releases/tag/v0.1.0
EOF

echo "=== 6/8: Ребрендинг текстов (README, CONTRIBUTING, KNOWN_LIMITATIONS, docs/*.md) ==="
REBRAND_FILES=("README.md" "CONTRIBUTING.md" "KNOWN_LIMITATIONS.md")
if [ -d docs ]; then
    while IFS= read -r -d '' f; do
        REBRAND_FILES+=("$f")
    done < <(find docs -name '*.md' -print0)
fi

for f in "${REBRAND_FILES[@]}"; do
    [ -f "$f" ] || continue
    sed -i.bak \
        -e "s/pip install ${OLD_DIST_NAME}/pip install ${NEW_DIST_NAME}/g" \
        -e "s#github\.com/${GITHUB_USER}/${OLD_REPO_PATH}#github.com/${GITHUB_USER}/${NEW_REPO_PATH}#g" \
        -e "s#git clone https://github.com/${GITHUB_USER}/${OLD_REPO_PATH}\.git#git clone https://github.com/${GITHUB_USER}/${NEW_REPO_PATH}.git#g" \
        -e "s#^cd ${OLD_REPO_PATH}\$#cd ${NEW_REPO_PATH}#" \
        -e "s#cd ${OLD_REPO_PATH}\$#cd ${NEW_REPO_PATH}#g" \
        -e "s/^# ${OLD_DIST_NAME}\$/# ${NEW_DIST_NAME}/" \
        -e "s/\*\*${OLD_DIST_NAME}\*\*/\*\*${NEW_DIST_NAME}\*\*/g" \
        -e "s/\bv0\.1\.1\b/v0.1.0/g" -e "s/\b0\.1\.1\b/0.1.0/g" \
        -e "s/\bv0\.1\.5\b/(planned, unreleased)/g" \
        -e "s#^${OLD_REPO_PATH}/\$#${NEW_REPO_PATH}/#" \
        "$f"
    rm -f "${f}.bak"
done

# Отдельно: project-layout дерево в README (строка с "gdn2-pallas/" внутри
# ```-блока, не URL и не "pip install"/"cd" -- ловим точечно).
if [ -f "README.md" ]; then
    sed -i.bak \
        -e "s#^${OLD_REPO_PATH}/\$#${NEW_REPO_PATH}/#" \
        "README.md"
    rm -f README.md.bak
fi

if [ -f "README.md" ]; then
    sed -i.bak \
        -e "s/@software{gdn2_pallas,/@software{atomic_ops,/" \
        -e "s/title  = {gdn2-pallas:/title  = {atomic_ops:/" \
        "README.md"
    rm -f README.md.bak
fi

echo "=== 7/8: Sanity-grep остаточных упоминаний старого имени ==="
echo "    (не считая исторических/контекстных упоминаний в docs -- см. вывод ниже)"
if grep -rn "${OLD_DIST_NAME}" --include="*.md" --include="*.toml" --include="*.py" . 2>/dev/null \
    | grep -v "^\./CHANGELOG.md" \
    | grep -v "ported from" \
    | grep -v "NVlabs"; then
    echo "[WARN] Найдены оставшиеся упоминания '${OLD_DIST_NAME}' выше -- проверь вручную."
else
    echo "    OK: явных упоминаний '${OLD_DIST_NAME}' вне CHANGELOG не найдено."
fi

echo "=== 8/8: git init + первый коммит ==="
git init -q
git add -A
git commit -q -m "v0.1.0 - initial public release (${NEW_DIST_NAME})

Fused Gated DeltaNet-2 (GDN-2) kernels for TPU v5e in JAX/Pallas.
Package renamed from ${OLD_DIST_NAME} to ${NEW_DIST_NAME} prior to first release.
Roadmap.md renamed to ROADMAP.md to match existing markdown links.
See CHANGELOG.md for details, ROADMAP.md for planned work,
KNOWN_LIMITATIONS.md for current limitations."

echo ""
echo "=================================================================="
echo "Локальный репозиторий готов: $NEW_DIR"
echo "  pip install ${NEW_DIST_NAME}   (после публикации на PyPI)"
echo "  версия: 0.1.0 (единственный релиз, без истории фикса B4)"
echo "  ROADMAP.md переименован из Roadmap.md (регистр приведён в порядок)"
echo ""

if [ "$CREATE_REMOTE" = "1" ]; then
    if command -v gh >/dev/null 2>&1; then
        echo "=== CREATE_REMOTE=1: создаю репозиторий на GitHub через gh CLI ==="
        gh repo create "${GITHUB_USER}/${NEW_REPO_PATH}" --public --source=. --remote=origin --push
        echo "Готово: https://github.com/${GITHUB_USER}/${NEW_REPO_PATH}"
    else
        echo "[WARN] CREATE_REMOTE=1, но 'gh' (GitHub CLI) не найден в PATH."
        echo "       Установи: https://cli.github.com/  затем 'gh auth login'"
        echo "       Либо создай репозиторий вручную на github.com и выполни:"
        echo "         git remote add origin https://github.com/${GITHUB_USER}/${NEW_REPO_PATH}.git"
        echo "         git branch -M main"
        echo "         git push -u origin main"
    fi
else
    echo "Репозиторий на GitHub НЕ создавался (это отдельный шаг)."
    echo "Чтобы создать и запушить вручную:"
    echo "  1) Создай пустой репозиторий '${NEW_REPO_PATH}' на github.com/${GITHUB_USER}"
    echo "  2) В папке $NEW_DIR выполни:"
    echo "       git remote add origin https://github.com/${GITHUB_USER}/${NEW_REPO_PATH}.git"
    echo "       git branch -M main"
    echo "       git push -u origin main"
    echo ""
    echo "  Или одной командой (нужен GitHub CLI 'gh' + 'gh auth login'):"
    echo "    CREATE_REMOTE=1 $0 \"$SRC_DIR\" \"$NEW_DIR\" \"$GITHUB_USER\" \"$NEW_REPO_PATH\""
fi
echo "=================================================================="

echo ""
echo "Проверь перед публикацией на PyPI (python -m build && twine upload):"
echo "  - pyproject.toml : name = \"${NEW_DIST_NAME}\", version = \"0.1.0\""
echo "  - atomic_ops/__init__.py : _version(\"${NEW_DIST_NAME}\")"
echo "  - README.md : pip install ${NEW_DIST_NAME}, cd ${NEW_REPO_PATH}, ссылки на github.com/${GITHUB_USER}/${NEW_REPO_PATH}"
echo "  - ROADMAP.md существует (не Roadmap.md) и ссылки [ROADMAP.md](ROADMAP.md) не битые"
echo "  - grep -rn '${OLD_DIST_NAME}' .   # должно не найти ничего, кроме контекстных упоминаний (NVlabs origin, CHANGELOG history)"
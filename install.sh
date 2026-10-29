#!/usr/bin/env bash
set -Eeuo pipefail
# Fduplex user-space installer: works inside a checkout or via curl | bash.
REPO="https://github.com/wanbnn/Fduplex.git"
REF="${FDUPLEX_REF:-main}"
DEST="${FDUPLEX_INSTALL_DIR:-$HOME/.local/share/fduplex}"
BIN="$HOME/.local/bin"
CONFIG="$HOME/.config/fduplex/env"
WORKSPACE="${FDUPLEX_WORKSPACE:-$HOME/Fduplex-workspace}"
say() { printf '[Fduplex] %s\n' "$*"; }
fail() { say "ERRO: $*" >&2; exit 1; }

command -v git >/dev/null || fail "Git não encontrado (sudo apt install git)"
command -v python3 >/dev/null || fail "Python 3 não encontrado"
python3 -c 'import sys; assert sys.version_info >= (3, 10)' || fail "Python 3.10+ necessário"

# Bash run from a checkout vs piped installer.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "$HERE/assistant.py" && -f "$HERE/requirements.txt" ]]; then
  APP="$HERE"
elif [[ -d "$DEST/.git" ]]; then
  APP="$DEST"
  say "Usando checkout existente em $APP (nenhum git pull automático)."
elif [[ -e "$DEST" ]]; then
  fail "Destino já existe e não é um repositório: $DEST"
else
  mkdir -p "$(dirname "$DEST")"
  say "Clonando o Fduplex..."
  git clone --depth=1 --branch "$REF" "$REPO" "$DEST"
  APP="$DEST"
fi

say "Preparando ambiente Python..."
if [[ ! -x "$APP/.venv/bin/python" ]]; then
  python3 -m venv "$APP/.venv" || fail "Falha ao criar venv. Instale python3-venv."
fi
"$APP/.venv/bin/python" -m pip install -r "$APP/requirements.txt"

mkdir -p "$BIN" "$HOME/.config/fduplex" "$WORKSPACE" "$HOME/Documents"
if [[ ! -f "$CONFIG" ]]; then
  {
    printf 'FDUPLEX_WORKSPACE=%q\n' "$WORKSPACE"
    printf 'FDUPLEX_READ_ROOTS=%q\n' "$HOME/Documents:$WORKSPACE"
    printf 'FDUPLEX_WRITE_ROOTS=%q\n' "$WORKSPACE"
    printf 'OPENAI_BASE_URL=%q\n' "http://localhost:1234/v1"
    printf 'OPENAI_API_KEY=%q\n' "local"
    printf 'OPENAI_MODEL=%q\n' "local-model"
  } > "$CONFIG"
  chmod 600 "$CONFIG"
fi

# Entrypoint stores full app path as an environment-safe shell assignment.
APP_QUOTED="$(printf '%q' "$APP")"
{
  printf '%s\n' '#!/usr/bin/env bash' 'set -Eeuo pipefail'
  printf 'APP=%s\n' "$APP_QUOTED"
  cat <<'LAUNCHER'
CONFIG="$HOME/.config/fduplex/env"
if [[ -f "$CONFIG" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG"
  set +a
fi
[[ -x "$APP/.venv/bin/python" ]] || { echo "Reinstale o Fduplex: ambiente Python ausente" >&2; exit 1; }
cd "$APP"
exec "$APP/.venv/bin/python" "$APP/assistant.py" "$@"
LAUNCHER
} > "$BIN/fduplex"
chmod 755 "$BIN/fduplex"

if [[ "${FDUPLEX_INSTALL_BROWSER:-1}" == "1" ]]; then
  say "Instalando Chromium (Playwright)..."
  "$APP/.venv/bin/python" -m playwright install chromium || say "Aviso: Chromium indisponível; somente a navegação MCP será afetada."
fi

# Persist PATH for new login shells without touching system profiles.
PROFILE="$HOME/.profile"
PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if ! grep -Fqx "$PATH_LINE" "$PROFILE" 2>/dev/null; then
  printf '\n# Fduplex launcher\n%s\n' "$PATH_LINE" >> "$PROFILE"
fi

say "Instalação concluída. Em um novo terminal, execute: fduplex"
say "Configuração: $CONFIG"
say "Requer um servidor de modelo compatível com OpenAI já ativo."
say "Para reconhecimento de voz, configure transcribe.cpp e GGUF conforme README.md."

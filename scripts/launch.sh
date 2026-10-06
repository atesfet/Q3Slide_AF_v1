#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"
mkdir -p .runtime/downloads .runtime/cache
printf '\nQ3Slide AF · preparing your local workspace\n\n'
CONDA_BIN=""
for candidate in "${CONDA_EXE:-}" "$APP_DIR/.runtime/miniforge/bin/conda" \
  "$(command -v conda || true)" "$HOME/miniforge3/bin/conda" \
  "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/anaconda3/bin/conda /opt/miniconda3/bin/conda; do
  if [[ "${Q3SLIDE_LOCAL_CONDA:-0}" == 1 && "$candidate" != "$APP_DIR/.runtime/miniforge/bin/conda" ]]; then continue; fi
  if [[ -n "$candidate" && -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
done
if [[ -z "$CONDA_BIN" ]]; then
  case "$(uname -s)" in Darwin) platform=MacOSX;; Linux) platform=Linux;; *) echo 'Use Start.bat on Windows.'; exit 1;; esac
  architecture="$(uname -m)"
  case "$platform:$architecture" in MacOSX:arm64|MacOSX:x86_64|Linux:x86_64|Linux:aarch64) ;; *) echo "Unsupported platform: $platform $architecture"; exit 1;; esac
  command -v curl >/dev/null || { echo 'curl is required to download the Conda installer. Install curl and launch again.'; exit 1; }
  version=26.7.2-0
  filename="Miniforge3-${version}-${platform}-${architecture}.sh"
  url="https://github.com/conda-forge/miniforge/releases/download/${version}/${filename}"
  installer="$APP_DIR/.runtime/downloads/$filename"
  echo 'Conda was not found. Downloading Miniforge into this application folder…'
  curl --fail --location --retry 3 --output "$installer" "$url"
  curl --fail --location --retry 3 --output "$installer.sha256" "$url.sha256"
  expected="$(awk '{print $1; exit}' "$installer.sha256")"
  if command -v shasum >/dev/null; then actual="$(shasum -a 256 "$installer" | awk '{print $1}')";
  else actual="$(sha256sum "$installer" | awk '{print $1}')"; fi
  [[ "$expected" =~ ^[a-fA-F0-9]{64}$ && "$actual" == "$expected" ]] || { echo 'Installer checksum verification failed. Please launch again.'; exit 1; }
  bash "$installer" -b -p "$APP_DIR/.runtime/miniforge"
  CONDA_BIN="$APP_DIR/.runtime/miniforge/bin/conda"
fi
CONDA_BASE="$("$CONDA_BIN" info --base)"
export MPLCONFIGDIR="$APP_DIR/.runtime/cache/matplotlib"
exec "$CONDA_BASE/bin/python" "$APP_DIR/scripts/bootstrap.py" --conda "$CONDA_BIN" "$@"

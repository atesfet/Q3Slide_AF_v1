#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if bash "$APP_DIR/scripts/launch.sh" "$@"; then
  exit 0
else
  result=$?
  if [[ -t 0 ]]; then read -r -p 'Startup failed. Press Return to close. ' reply; fi
  exit "$result"
fi

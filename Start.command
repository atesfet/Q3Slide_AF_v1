#!/bin/bash
cd "$(dirname "$0")" || exit 1
bash scripts/launch.sh "$@"
result=$?
if [ "$result" -ne 0 ]; then
  printf '\nStartup failed. The explanation is above. Press Return to close.\n'
  read -r reply
fi
exit "$result"

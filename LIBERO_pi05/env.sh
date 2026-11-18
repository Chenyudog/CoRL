#!/usr/bin/env bash
# In bash: source env.sh
_pi05_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_pi05_workspace/.venv/bin/activate"
export OPENPI_DATA_HOME="$_pi05_workspace/.cache/openpi"
export LIBERO_CONFIG_PATH="$_pi05_workspace/.libero-config"
unset _pi05_workspace

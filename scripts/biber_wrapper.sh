#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Temporary workaround for environments where biber (PAR) fails to find Unicode::UCD's unicore/version.
# The PAR extraction directory is pinned under the project, and the version file is patched in when needed.

export PAR_GLOBAL_TEMP="${ROOT}/build/par"
mkdir -p "$PAR_GLOBAL_TEMP"

find_unicore_version() {
  local candidates=(
    "/System/Library/Perl/5.34/unicore/version"
    "/System/Library/Perl/5.36/unicore/version"
    "/System/Library/Perl/5.38/unicore/version"
  )

  local perl_priv=""
  if perl_priv="$(perl -MConfig -e 'print $Config{privlibexp}' 2>/dev/null)"; then
    candidates+=("${perl_priv}/unicore/version")
  fi

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done

  return 1
}

run_biber() {
  local out
  out="$(mktemp "${ROOT}/build/biber.XXXXXX.log")"
  if biber "$@" >"$out" 2>&1; then
    cat "$out"
    rm -f "$out"
    return 0
  fi
  cat "$out"
  rm -f "$out"
  return 2
}

if run_biber "$@"; then
  exit 0
fi

UNICORE_VERSION_SRC=""
if UNICORE_VERSION_SRC="$(find_unicore_version 2>/dev/null)"; then
  mapfile -t targets < <(find "$PAR_GLOBAL_TEMP" -type d -path '*/inc/lib/unicore' 2>/dev/null || true)
  if (( ${#targets[@]} > 0 )); then
    for d in "${targets[@]}"; do
      if [[ ! -f "${d}/version" ]]; then
        cp "$UNICORE_VERSION_SRC" "${d}/version"
      fi
    done
    run_biber "$@" && exit 0
  fi
fi

exit 2

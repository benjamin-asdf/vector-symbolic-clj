#!/usr/bin/env bash
# Run experiment specs one after another: scripts/run-specs.sh WORKERS spec1 [spec2 ...]
# Each spec's runner log goes to out/<name>.runlog.
set -u
cd "$(dirname "$0")/.."
workers=$1
shift
for spec in "$@"; do
  name=$(basename "$spec" .edn)
  echo "$(date +%T) start $name ($workers workers, $(free -g | awk '/Mem:/ {print $7}') GB available)"
  clojure -M:jvm:exp "$spec" --workers "$workers" > "out/$name.runlog" 2>&1
  echo "$(date +%T) done $name: $(tail -1 "out/$name.runlog")"
done

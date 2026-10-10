#!/usr/bin/env bash
# Finish the full cache (resumable, retried), wait for free VRAM, then train (resumable).
#   ./run_generalist.sh [run-config]     default: generalist_v3
# Logs: out/cache_main.log, out/<run-config>.log.
cd "$(dirname "$0")"
RUN="${1:-generalist_v3}"
mkdir -p out
count() { wc -l < data/cache_main/index.jsonl 2>/dev/null || echo 0; }
for attempt in $(seq 1 20); do
  before=$(count)
  .venv/bin/irisedit cache data 2>&1 | grep --line-buffered -v -i "warn\|malformed" | tee -a out/cache_main.log
  after=$(count)
  echo "[launcher] cache attempt $attempt: $before -> $after pairs ($(date +%T))" | tee -a out/cache_main.log
  [ "$after" -eq "$before" ] && break
done
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 20000 ]; do
  echo "[launcher] waiting for 20GB free VRAM ($(date +%T))" >> "out/$RUN.log"; sleep 120
done
.venv/bin/irisedit train "$RUN" 2>&1 | grep --line-buffered -v -i warn | tee -a "out/$RUN.log"

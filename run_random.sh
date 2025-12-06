#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""   # force CPU

OUT_DIR="random_results"


MAXJOBS="${MAXJOBS:-12}"   # override with environment variable if desired
# use 0..999 for 1000 seeds; adjust end as needed
for SEED in {0..9999}; do
  # throttle: wait while running background jobs >= MAXJOBS
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do
    sleep 0.5
  done

  echo "Launching seed $SEED"

  python pet.py \
      random_original \
      "$SEED" \
      "$OUT_DIR" \
      > /dev/null 2>&1 &
done

wait
echo "All seeds finished."
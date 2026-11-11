#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")"

common_args=(
  Si
  --sc 1 1 1
  --k 1 1 1
  --ansatz au-fixed
  --nparam 4
  --paulis YZZ YIZ YZI YII
  --fixed-occ
  --qpu
  --dev ionq.simulator
  --shots 1000
  --group
  --max-itr 500
)

for run in 01 02 03 04; do
  log="data/Si_ionq_simulator_fresh${run}.log"
  nohup env \
    MPLCONFIGDIR=/tmp/matplotlib-cache \
    JAX_PLATFORMS=cpu \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    python3 -u qudft.py "${common_args[@]}" \
    >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"${log%.log}.pid"
  printf 'fresh%s pid=%s log=%s\n' "$run" "$pid" "$log"
done

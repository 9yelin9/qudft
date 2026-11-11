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
  --grouping gc
  --tol 1e-4
  --max-itr 500
)

checkpoints=(
  data/Si_sc1x1x1_k1x1x1_qudft_au-fixed_P4_9b32cd20_fixed-occ_ionq-simulator_shots1000_rep03.out
  data/Si_sc1x1x1_k1x1x1_qudft_au-fixed_P4_9b32cd20_fixed-occ_ionq-simulator_shots1000_rep02.out
  data/Si_sc1x1x1_k1x1x1_qudft_au-fixed_P4_9b32cd20_fixed-occ_ionq-simulator_shots1000_rep01.out
  data/Si_sc1x1x1_k1x1x1_qudft_au-fixed_P4_9b32cd20_fixed-occ_ionq-simulator_shots1000.out
)

for index in 0 1 2 3; do
  run=$(printf '%02d' "$((index + 1))")
  checkpoint=${checkpoints[$index]}
  log="data/Si_ionq_simulator_resume${run}.log"

  if [[ ! -f "$checkpoint" ]]; then
    printf 'missing checkpoint: %s\n' "$checkpoint" >&2
    exit 1
  fi

  nohup env \
    MPLCONFIGDIR=/tmp/matplotlib-cache \
    JAX_PLATFORMS=cpu \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    python3 -u qudft.py "${common_args[@]}" \
      --resume "$checkpoint" \
      >"$log" 2>&1 </dev/null &

  pid=$!
  printf '%s\n' "$pid" >"${log%.log}.pid"
  printf 'resume%s pid=%s checkpoint=%s log=%s\n' \
    "$run" "$pid" "$checkpoint" "$log"
done

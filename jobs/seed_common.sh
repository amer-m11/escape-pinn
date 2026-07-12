#!/bin/bash
# Shared seed-sweep body: sourced by a SLURM array job (see
# jobs/seed_template.sh) after it exports LABEL, CASE and SETS.
#
# Each array task runs one seed (= $SLURM_ARRAY_TASK_ID) of the case into its
# own per-seed run dir, so one `#SBATCH --array=0-4` submission produces the
# five single-seed runs used for a mean±std accuracy estimate.
set -euo pipefail

: "${LABEL:?seed_common.sh: LABEL must be exported by the wrapper}"
: "${CASE:?seed_common.sh: CASE must be exported by the wrapper}"
: "${SETS:?seed_common.sh: SETS must be exported by the wrapper}"

# SEED = array task id (per-task single seed). Falls back to 0 for a bare run.
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
JOBTAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
RUN_DIR="output/pinn/seed_${LABEL}/$(date +%Y%m%d)_${JOBTAG}_seed${SEED}"
mkdir -p output/logs "$RUN_DIR"
echo "Date: $(date)"; echo "Config: ${LABEL}  Seed: ${SEED}  Run dir: ${RUN_DIR}"

python -m pipenv run python -m pinn_solver.run \
    --config jobs/config.yaml --case "$CASE" \
    $SETS --set seed=${SEED} \
    --output-dir "$RUN_DIR"

echo "Seed ${SEED} of ${LABEL} completed."; echo "Date: $(date)"

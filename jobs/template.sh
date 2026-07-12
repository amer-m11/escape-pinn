#!/bin/bash

#SBATCH --job-name=Template
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --qos=l40s-6hours
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output/logs/pinn-%j.log
#SBATCH --error=output/logs/pinn-%j.log

# ------------------------------------------------------------------
# SLURM template: train one PINN case from jobs/config.yaml.
#
# Edit case (a top-level key of jobs/config.yaml. The `template` case
# documents every option) and add `--set` overrides as needed. Submit
# from a local machine with bin/sbatch, or directly on the cluster:
#
#     bin/sbatch jobs/template.sh          # over SSH, from local
#     sbatch jobs/template.sh              # on the cluster
#
# The run directory receives metrics.json, model.pt and the three
# diagnostic figures (see readme.md "Quick start").
# ------------------------------------------------------------------

set -euo pipefail

set +u; source /etc/profile; set -u  # initialise lmod
module load CUDA/13.0.0
module load Python/3.14.2-GCCcore-15.2.0
export PYTHONUNBUFFERED=1

CASE="gaussian_lens_cart_3d"

RUN_DIR="output/pinn/${CASE}/$(date +%Y%m%d)_${SLURM_JOB_ID}"
mkdir -p output/logs "$RUN_DIR"

echo "Date: $(date)"
echo "Running job $SLURM_JOB_ID on $(hostname)"
echo "Run dir: $RUN_DIR"

python -m pipenv run python -m pinn_solver.run \
    --config jobs/config.yaml --case "$CASE" \
    --output-dir "$RUN_DIR"

# Mesh reference solver alternative (CPU-bound):
#   python -m pipenv run python -m mesh_algorithm.examples.variable_medium_3d \
#       --output-dir "$RUN_DIR" --n-x1 41 --n-x2 41 --n-x3 41 --n-phi 16 --n-theta 32

echo "Job $SLURM_JOB_ID completed."
echo "Date: $(date)"

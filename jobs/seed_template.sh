#!/bin/bash

#SBATCH --job-name=SeedSweep
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --qos=rtx4090-6hours
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output/logs/seed-%A_%a.log
#SBATCH --error=output/logs/seed-%A_%a.log

# ------------------------------------------------------------------
# SLURM array template: the same case trained under X seeds.
#
# Submit: bin/sbatch jobs/seed_template.sh
# ------------------------------------------------------------------

set -euo pipefail
set +u; source /etc/profile; set -u
module load CUDA/13.0.0
module load Python/3.14.2-GCCcore-15.2.0
export PYTHONUNBUFFERED=1

export LABEL=cart_champ
export CASE=gaussian_lens_cart_3d
export SETS="--set net.trunk=pirate \
      --set net.exit_time_features=true \
      --set net.factored_eikonal=true \
      --set net.per_channel_heads=true \
      --set net.arch_name=fc-pirate-cart-heads-3d"

source jobs/seed_common.sh

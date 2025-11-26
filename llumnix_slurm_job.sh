#!/bin/bash
#SBATCH --job-name=llumnix_slurm_job
#SBATCH --partition=savio4_htc
#SBATCH --account=fc_cosi
#SBATCH --time=02:00:00
#SBATCH --array=0-59
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err

mkdir -p logs

module load python/3.10

source .venv/bin/activate

echo "HOST=$(hostname) JOBID=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID" >&2

python3 run_tests.py --index "$SLURM_ARRAY_TASK_ID"
#!/bin/bash
#SBATCH --job-name=results
#SBATCH --account=csd969
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=96G
#SBATCH --gpus=2
#SBATCH --time=00:30:00
#SBATCH --output=script_logs/%x.o%j.txt
#SBATCH --error=script_logs/%x.e%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --mail-user=etwatson@ucsd.edu

module purge
module load slurm
module load gpu
module load gcc/10.2.0
module load cuda/11.2.2
module load cudnn/8.1.1.33-11.2

echo "Activating virtual environment..."
source /home/ewatson/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda shell"; exit 1; }
conda activate leakless || { echo "Failed to activate virtual environment"; exit 1; }

CKPT=${1:?Usage: sbatch $0 <checkpoint.ckpt>}

echo "Starting Python script with checkpoint '$CKPT'..."
srun --unbuffered python results.py -c "$CKPT" -n 5000|| { echo "Python script failed"; exit 1; }

echo "Job completed."
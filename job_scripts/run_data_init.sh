#!/bin/bash
#SBATCH --job-name=data_init
#SBATCH --account=csd969
#SBATCH --partition=shared
#SBATCH --constraint="lustre"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=script_logs/%x.o%j.txt
#SBATCH --error=script_logs/%x.e%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=etwatson@ucsd.edu

module purge
module load slurm
module load cpu/0.15.4
module load gcc/10.2.0

echo "Activating virtual environment..."
source /home/ewatson/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda shell"; exit 1; }
conda activate leakless || { echo "Failed to activate virtual environment"; exit 1; }

echo "Starting Python script..."
srun --unbuffered python data_init.py || { echo "Python script failed"; exit 1; }

echo "Job completed."

#!/bin/bash

#SBATCH --partition=highgpu
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=64g
#SBATCH --job-name=moe_bitflip
#SBATCH --error=moe_bitflip_error
#SBATCH --output=moe_bitflip_output

module load python
module load anaconda
module load cuda

conda activate security_project

python project.py
python figure.py
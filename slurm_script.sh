#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=64g
#SBATCH --job-name=moe_bitflip
#SBATCH --error=moe_bitflip
#SBATCH --output=moe_bitflip

module load python
module load anaconda
module load cuda

python project.py
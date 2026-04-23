#SBATCH --account=bebv-delta-gpu
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:gpuH100:1
#SBATCH --mem=64g
#SBATCH --job-name=moe_bitflip
#SBATCH --error=moe_bitflip
#SBATCH --output=moe_bitflip

module load python
module load anaconda3_gpu
module load cuda

python project.py
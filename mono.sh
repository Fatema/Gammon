#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -p "ug-gpu-small"
#SBATCH --qos="short"
#SBATCH -t 2-0

source /etc/profile

module load cuda/9.0-cudnn7.0

source ~/.venv/tf/bin/activate

python3 main.py -mono

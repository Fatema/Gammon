#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH -p "cpu"
#SBATCH --qos="long-cpu"
#SBATCH -t 14-0

source /etc/profile

source ~/.venv/tf-cpu/bin/activate

python3 main.py

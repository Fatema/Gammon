#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -p "cpu"
#SBATCH --qos="long-cpu"

source /etc/profile

source ~/.venv/tf-cpu/bin/activate

python3 main.py

#!/bin/bash
#SBATCH --mem-per-cpu=2048M
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1

dataset=$1

python main.py --dataset $dataset --run_number 1  --epoch_number 100 --out_of_sample --fine_tune --fine_tune_number 100

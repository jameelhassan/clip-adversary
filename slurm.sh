#!/bin/bash

#SBATCH --job-name=ViTB16_1e-3_CIFAR10_MinCorr5_encCLIPfreeze_Cosine_eps01             # Job name
#SBATCH --output=./cluster_out/ViTB16_1e-3_CIFAR10__MinCorr5_encCLIPfreeze_Cosine_eps01.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

hostname
python train.py
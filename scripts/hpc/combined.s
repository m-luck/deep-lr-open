#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --time=1-0
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name=gpt2_lr
#SBATCH --mail-user mll469@nyu.edu
#SBATCH --mail-type=END
#SBATCH --output=outfull_comb%j

source /scratch/mll469/env_train/bin/activate

cd /scratch/mll469/gpt2deeplr

python gpt2train.py --base_dir . --input_type WORDS --num_workers 3 --cache_in_ram

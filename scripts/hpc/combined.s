#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --time=1-0
#SBATCH --gres=gpu:k80:1

#SBATCH --job-name=deeplr_w_gpt2
#SBATCH --mail-user mll469@nyu.edu
#SBATCH --mail-type=END
#SBATCH --output=out_com%j

source /scratch/mll469/env_train/bin/activate

cd /scratch/mll469/gpt2deeplr

python gpt2train.py --base_dir . --num_workers 3

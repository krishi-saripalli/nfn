#!/bin/bash

#SBATCH -J train_nfn_single #name of the job
#SBATCH --nodes=1                          # number of nodes to reserve
#SBATCH -n 8                   # number of CPU cores
#SBATCH --mem=100gb                          # memory per node
#SBATCH -t 96:00:00                         
#SBATCH --partition=3090-gcondo --gres=gpu:1 # partition and number of GPUs (per node)
#SBATCH --export=CXX=g++                    # compiler
#SBATCH -o /users/ksaripal/logs/train_nfn_single.out   #out logs
#SBATCH -e /users/ksaripal/logs/train_nfn_single.err   #error logs


cd /users/ksaripal/BVC/nfn/

module load anaconda/2023.09-0-7nso27y
source /gpfs/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh
conda activate nfn

python -m experiments.launch_inr2array dset=mnist model=nft compile=false

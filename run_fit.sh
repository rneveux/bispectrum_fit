#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=10G
#SBATCH --constraint=avx

srun python3 bispectrum/fit/fit.py -config $1

#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=10G
#SBATCH --constraint=avx

srun python3 bispectrum/fit/fit_test.py -config $1

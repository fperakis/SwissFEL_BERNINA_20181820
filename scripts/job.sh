#!/bin/bash
#
#SBATCH --job-name=process
#SBATCH --output=job_output.txt

srun  ./process.py -r 0018_droplets_10um_2mm -s 1000
srun  ./process.py -r 0019_droplets_10um_2mm -s 1000
srun  ./process.py -r 0020_droplets_10um_2mm -s 1000
srun  ./process.py -r 0021_droplets_10um_2mm -s 1000
srun  ./process.py -r 0022_droplets_10um_2mm -s 1000

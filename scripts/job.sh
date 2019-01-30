#!/bin/bash
#
#SBATCH --job-name=process
#SBATCH --output=job_output.txt

srun  ./process.py -r 0024_droplets_10um_4mm &
srun  ./process.py -r 0025_droplets_10um_4mm &
srun  ./process.py -r 0026_droplets_10um_4mm &
srun  ./process.py -r 0027_droplets_10um_4mm &
srun  ./process.py -r 0028_droplets_10um_4mm &
srun  ./process.py -r 0029_droplets_10um_4mm &
srun  ./process.py -r 0030_droplets_10um_4mm &
srun  ./process.py -r 0031_droplets_10um_4mm &
srun  ./process.py -r 0032_droplets_10um_4mm 

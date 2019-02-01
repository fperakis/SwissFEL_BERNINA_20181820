#!/bin/bash
#
#SBATCH --job-name=process
#SBATCH --output=job_output.txt

srun  ./process.py -r 0024_droplets_10um_4mm &


#!/bin/bash
#
#SBATCH --job-name=process
#SBATCH --output=job_output.txt

srun  ./process.py -r 0000_test01 -s 10

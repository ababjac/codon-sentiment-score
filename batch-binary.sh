#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 
#SBATCH --time=12:00:00
#SBATCH -e ./jobs/myjob.e%j
#SBATCH -o ./jobs/myjob.o%j 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ababjac@vols.utk.edu

cd $SLURM_SUBMIT_DIR
source $SCRATCHDIR/pyvenv/bin/activate
python scripts/train_binary.py


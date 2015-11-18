#!/bin/bash
# parallel job using 48 cores. and runs for 4 hours (max)
#SBATCH -A chan
#SBATCH -N 1 # node count
#SBATCH -t 100:00:00
# sends mail when process begins, and 
# when it ends. Make sure you define your email 
# address.
#SBATCH --mail-type=end
#SBATCH --mail-user=zhendong@princeton.edu
#SBATCH --ntasks-per-node=10
#SBATCH --mem=30000
#SBATCH -C ivy
# problems with mpiexec/mpirun so use srun
export SCRATCHDIR1="/scratch/gpfs/zhendong/cas_hs1"
mkdir -p $SCRATCHDIR1
export OMP_NUM_THREADS=10
python -u local.py > local.out 

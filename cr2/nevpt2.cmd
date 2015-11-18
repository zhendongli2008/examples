#!/bin/bash
# parallel job using 48 cores. and runs for 4 hours (max)
#SBATCH -A chan
#SBATCH -N 1   # node count
#SBATCH -t 100:00:00
# sends mail when process begins, and 
# when it ends. Make sure you define your email 
# address.
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=zhendong@princeton.edu
#SBATCH --cpus-per-task=10
#SBATCH -C ivy
cd /tigress/zhendong/cal-femoco/dimer/nevpt2
# problems with mpiexec/mpirun so use srun
module purge
module add intel-mkl/11.2/2/64
module add h5py27/2.2.1/hdf5-1.8.12                            
module add hdf5/gcc/1.8.12                                     
module add pyscf
module add python/2.7
export OMP_NUM_THREADS=10
python cr2-scan.py

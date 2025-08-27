#!/bin/bash
#SBATCH -J test_core1
#SBATCH -p batch
#SBATCH -w cpu04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time 00:30:00


# make clean
# make all

nproc=8
echo "mpirun -np $nproc ./core.exe"
mpirun -np $nproc ./core.exe

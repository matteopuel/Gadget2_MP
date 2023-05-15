#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-cline
#SBATCH --ntasks=32

module add python/2.7.14 gsl/2.3 openmpi/2.1.1 hdf5/1.8.18 nixpkgs/16.09 gtk+3/3.20.9 intel/2016.4 fftw-mpi/2.1.5

cd /home/mpuel/projects/def-cline/mpuel/ADMgal/Gadget2_MP

mpirun -np 32 ./Gadget2 ADMmodel.param

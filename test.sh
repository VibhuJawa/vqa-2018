#!/bin/bash
#SBATCH -n 4
#SBATCH -p gpu 
#SBATCH -t 01:0:0

module load cuda/9.0
#module load singularity/2.4
python test_dataloader.py
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
#export SINGULARITY_HOME=$PWD:/home/$USER

#singularity pull --name pytorch.simg shub://marcc-hpc/pytorch
#singularity exec --nv ./pytorch.simg python y


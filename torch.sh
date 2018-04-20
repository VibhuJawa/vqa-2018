#!/bin/bash
#SBATCH -N 4
#SBATCH -n 24
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -t 04:55:0

module load cuda/9.0
#module load singularity/2.4

python test.py
python extract_image_features.py --data_split train2014
python extract_image_features.py --data_split val2014
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
#export SINGULARITY_HOME=$PWD:/home/$USER

#singularity pull --name pytorch.simg shub://marcc-hpc/pytorch
#singularity exec --nv ./pytorch.simg python y


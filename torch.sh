#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0:5:0

module load cuda/9.0
#module load singularity/2.4

python test.py
python main.py --num-workers 8
#python extract_image_features.py --data_split train2014
#python extract_image_features.py --data_split val2014
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
#export SINGULARITY_HOME=$PWD:/home/$USER

#singularity pull --name pytorch.simg shub://marcc-hpc/pytorch
#singularity exec --nv ./pytorch.simg python y


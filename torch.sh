#!/bin/bash
#SBATCH -n 48 
#SBATCH -p parallel
#SBATCH -t 8:00:00

module load cuda/9.0
#module load singularity/2.4

python test.py
python main.py --num-workers 1
#python extract_image_features.py --data_split train2014
#python extract_image_features.py --data_split val2014
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
#export SINGULARITY_HOME=$PWD:/home/$USER

#singularity pull --name pytorch.simg shub://marcc-hpc/pytorch
#singularity exec --nv ./pytorch.simg python y


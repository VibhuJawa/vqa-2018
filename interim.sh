#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 2
#SBATCH -p shared
#SBATCH -t 0:20:0

#module load cuda/9.0
#module load singularity/2.4
python extract_annotations_interim.py --folder /home-3/pmahaja2@jhu.edu/scratch/vqa2018_data/
python extract_annotations_processed.py --dir /home-3/pmahaja2@jhu.edu/scratch/vqa2018_data/
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
#export SINGULARITY_HOME=$PWD:/home/$USER

#singularity pull --name pytorch.simg shub://marcc-hpc/pytorch
#singularity exec --nv ./pytorch.simg python y


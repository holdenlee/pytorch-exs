#!/bin/bash

#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8000

eval "$(pyenv virtualenv-init -)"

cd ~/code/basics/

module load cudatoolkit/9.0 cudnn/cuda-9.0/7.0.3
pyenv activate pt
#python basics.py | tee mnist_log.txt
python mnist_train.py --data-dir /home/holdenl/data/mnist --save-file /home/holdenl/models/mnist --download False | tee mnist_log.txt
python mnist_reload.py --data-dir /home/holdenl/data/mnist --save-file /home/holdenl/models/mnist --download False | tee mnist_log2.txt

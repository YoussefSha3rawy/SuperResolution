#!/bin/bash
#SBATCH --job-name inm705_SR                      # Job name
#SBATCH --partition=prigpu                        # Select the correct partition.
#SBATCH --nodes=1                                # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=4                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --mem=90GB                                 # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=48:00:00                            # Expected ammount of time to run Time limit hrs:min:sec
#SBATCH --gres=gpu:1                               # Use one gpu.
#SBATCH -e results/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o results/%x_%j.o                         # [%x with the job name], make sure 'results' folder exists.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adfx751@city.ac.uk

#Enable modules command

source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu
export WANDB_API_KEY=37d31add06ffd6210d871e1462ad8777b14e5999

echo $WANDB_API_KEY


python --version
#module load libs/nvidia-cuda/11.2.0/bin
nvidia-smi

wandb login $WANDB_API_KEY --relogin
#pip freeze
#Run your script.
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
python3 train.py --config configs/SRResNet.yaml

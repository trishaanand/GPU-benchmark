#!/bin/sh
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH -C TitanX-Pascal
#SBATCH --gres=gpu:1

. /etc/bashrc
. /etc/profile.d/modules.sh

module load opencl-nvidia/10.0

./pagerank /var/scratch/alvarban/BSc_2k19/graphs/G500/graph500-23.e /var/scratch/alvarban/BSc_2k19/graphs/G500/graph500-23.v

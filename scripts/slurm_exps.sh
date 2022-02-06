#!/bin/sh

#SBATCH -J array
#SBATCH -N 1
#SBATCH --array=1-6
#SBATCH --output=slurm-%j-%A-%a.out
#SBATCH --error=slurm-%j-%A-%a.err

module purge
module load go/1.13.15 singularity/3.8.3

source ./init_exp_list.sh   # load list of args

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
aa=${args[$SLURM_ARRAY_TASK_ID]}
image="${DIR}/../reward_shaping.sif"

echo "Dir: ${DIR}"
echo "Image SIF: ${image}"
echo "ARGS: ${aa}"
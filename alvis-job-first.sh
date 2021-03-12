#!/bin/bash
#SBATCH -J qec-test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:05:00
#SBATCH -A SNIC2020-33-2 -p alvis
#SBATCH --gpus-per-node=V100:1
#SBATCH --output=./logs-sbatch/logs-%j.out

echo "Starting job on cluster"

SINGULARITY_IMAGE_NAME=qec-mp.sif
DOCKER_IMAGE_NAME=docker://xero32/qec-mp:first

WORKDIR=surface-rl-decoder
LOG_PATH=${HOME}/${WORKDIR}/tmp_runs

mkdir -p ${LOG_PATH}

job_stats.py ${SLURM_JOB_ID}

singularity run \
    -B ${LOG_PATH}:/${WORKDIR}/runs:rw \
    docker://xero32/qec-mp:first \
    /bin/bash -c \
    "cd /${WORKDIR}; \
    python --version; \
    python /${WORKDIR}/src/distributed/start_distributed_mp.py"

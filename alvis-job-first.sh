#!/bin/bash
#SBATCH -J qec-test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:00:40
#SBATCH -A SNIC2020-33-2 -p alvis
#SBATCH --gpus-per-node=V100:1
#SBATCH --output=./logs-sbatch/logs-%j.out

echo "###### Starting job on cluster"
echo "Learner device: ${DISTRIBUTED_CONFIG_LEARNER_DEVICE}"
echo "System size: ${CONFIG_ENV_SIZE}"

IMAGE_WORKDIR=surface-rl-decoder
CLUSTER_WORKDIR=surface-rl-decoder
LOG_PATH=${HOME}/${CLUSTER_WORKDIR}/tmp_runs

SINGULARITY_IMAGE_NAME=${CLUSTER_WORKDIR}/qec-mp.sif
DOCKER_IMAGE_NAME=docker://xero32/qec-mp:first

mkdir -p ${LOG_PATH}

job_stats.py ${SLURM_JOB_ID}

singularity run --nv \
    -B ${LOG_PATH}:/${IMAGE_WORKDIR}/runs:rw \
    --env-file ${CLUSTER_WORKDIR}/conf.env \
    ${SINGULARITY_IMAGE_NAME} \
    /bin/bash -c \
    "cd /${IMAGE_WORKDIR}; \
    python --version; \
    nvidia-smi;
    python -c 'import torch; print(torch.cuda.is_available())'; \
    python -c 'import torch.cuda as tc; id = tc.current_device(); print(tc.get_device_name(id))'; \
    python /${IMAGE_WORKDIR}/src/distributed/start_distributed_mp.py"

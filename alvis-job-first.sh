#!/bin/bash
#SBATCH -J qec-test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:00:20
#SBATCH -A SNIC2020-33-2 -p alvis
#SBATCH --gpus-per-node=V100:1
#SBATCH --output=./logs-sbatch/logs-%j.out

echo "###### Starting job on cluster"
echo "Learner device: ${DISTRIBUTED_CONFIG_LEARNER_DEVICE}"
echo "System size: ${CONFIG_ENV_SIZE}"

IMAGE_WORKDIR=surface-rl-decoder
CLUSTER_WORKDIR=surface-rl-decoder
TENSORBOARD_PATH_SPECIFICATION=tmp_runs
NETWORK_SAVE_PATH_CLUSTER=tmp_networks
LOG_PATH_SPECIFICATION_CLUSTER=tmp_runs
NETWORK_SAVE_PATH_IMAGE=networks
LOG_PATH_SPECIFICATION_IMAGE=runs


if [ -z "${SLURM_JOB_ID}" ]
    then
    echo "No Slurm Job ID found. Revert to local testing mode."
    CLUSTER_WORKDIR=$(pwd)
    LOG_PATH=${CLUSTER_WORKDIR}/${LOG_PATH_SPECIFICATION_CLUSTER}
    NETWORK_PATH=${CLUSTER_WORKDIR}/${NETWORK_SAVE_PATH_CLUSTER}
else
    echo "Slurm Job ID is ${SLURM_JOB_ID}." 
    job_stats.py ${SLURM_JOB_ID}
    LOG_PATH=${HOME}/${CLUSTER_WORKDIR}/${LOG_PATH_SPECIFICATION_CLUSTER}
    NETWORK_PATH=${HOME}/${CLUSTER_WORKDIR}/${NETWORK_SAVE_PATH_CLUSTER}
fi

if [ -z "$2" ]
then
    SINGULARITY_IMAGE_NAME=${CLUSTER_WORKDIR}/qec-mp.sif
    DOCKER_IMAGE_NAME=docker://xero32/qec-mp:first
else
    SINGULARITY_IMAGE_NAME=${CLUSTER_WORKDIR}/qec-mp_$2.sif
    DOCKER_IMAGE_NAME=docker://xero32/qec-mp:$2
fi

mkdir -p ${LOG_PATH}
mkdir -p ${NETWORK_PATH}


if [ -z "$1" ]
    then
    echo ""
    echo "No argument for env-config file supplied!"
    echo "Resort to default values"
    echo ""
    echo "Run on image ${SINGULARITY_IMAGE_NAME}"
    echo ""

    singularity run --nv \
        -B ${LOG_PATH}:/${IMAGE_WORKDIR}/${LOG_PATH_SPECIFICATION_IMAGE}:rw \
        -B ${NETWORK_PATH}:/${IMAGE_WORKDIR}/${NETWORK_SAVE_PATH_IMAGE}:rw \
        ${SINGULARITY_IMAGE_NAME} \
        /bin/bash -c \
        "cd /${IMAGE_WORKDIR}; \
        python --version; \
        nvidia-smi;
        python -c 'import torch; print(torch.cuda.is_available())'; \
        python -c 'import torch.cuda as tc; id = tc.current_device(); print(tc.get_device_name(id))'; \
        python /${IMAGE_WORKDIR}/src/distributed/start_distributed_mp.py"
else
    echo ""
    echo "Use config file $1"
    echo ""
    echo "Run on image ${SINGULARITY_IMAGE_NAME}"
    echo ""

    singularity run --nv \
        -B ${LOG_PATH}:/${IMAGE_WORKDIR}/runs:rw \
        -B ${NETWORK_PATH}:/${IMAGE_WORKDIR}/networks:rw \
        --env-file $1 \
        ${SINGULARITY_IMAGE_NAME} \
        /bin/bash -c \
        "cd /${IMAGE_WORKDIR}; \
        python --version; \
        nvidia-smi;
        python -c 'import torch; print(torch.cuda.is_available())'; \
        python -c 'import torch.cuda as tc; id = tc.current_device(); print(tc.get_device_name(id))'; \
        python /${IMAGE_WORKDIR}/src/distributed/start_distributed_mp.py"
fi
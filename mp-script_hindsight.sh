#!/bin/bash
#SBATCH -J qec-test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:10:00
#SBATCH -A SNIC2020-33-2 -p alvis
#SBATCH --gpus-per-node=V100:1
#SBATCH --output=./logs-sbatch/logs-%j.out

#####################################################################
# This is an extended version of our
# job script to deploy a multiprocessing
# deep reinforcement learning system 
# on Alvis.
#
# credits to https://gist.github.com/rosterloh/1bff516dfcdd8573fde0
#
#####################################################################

IMAGE_WORKDIR=/surface-rl-decoder
CLUSTER_WORKDIR=surface-rl-decoder

NETWORK_PATH_IMAGE=networks
TENSORBOARD_PATH_IMAGE=runs
NETWORK_PATH_CLUSTER=tmp_networks
TENSORBOARD_PATH_CLUSTER=tmp_runs
DEPLOY_DATE=$(date '+%Y-%m-%d_%H:%M:%S')
PROGRAM_PATH=src/distributed/start_distributed_mp.py
JOB_DESCRIPTION=

if [ -z ${SLURM_JOB_ID} ]; then
    SAVE_INFO_PATH=${DEPLOY_DATE}
    CLUSTER_WORKDIR=$(pwd)
else
    SAVE_INFO_PATH=${SLURM_JOB_ID}
    CLUSTER_WORKDIR=${HOME}/${CLUSTER_WORKDIR}
fi

IMAGE_NAME=qec-mp.sif
PATH_TO_IMAGE=${CLUSTER_WORKDIR}/${IMAGE_NAME}

echo "SAVE_INFO_PATH = ${SAVE_INFO_PATH}"
#Set fonts for Help.
NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`

#Set Script Name variable
SCRIPT=`basename ${BASH_SOURCE[0]}`
function HELP {
    echo -e \\n"Usage guide for ${SCRIPT}"\\n
    echo -e "Basic usage: sbatch ${SCRIPT}"
    echo -e "to schedule a run on the cluster."\\n
    echo -e "The following command line switches are recognized:"
    echo "${REV}-d${NORM}  --Description of the current run. Required field."
    echo "${REV}-c${NORM}  --Pass an environment configuration file. Looks for the config flie in the working directory."
    echo "${REV}-C${NORM}  --Pass an environment configuration file. Expects full path."
    echo "${REV}-i${NORM}  --Define the name of the image to be used. Default is ${BOLD}${IMAGE_NAME}${NORM}"
    echo "${REV}-I${NORM}  --Define the absolute path to the image being used. Default is ${BOLD}${PATH_TO_IMAGE}${NORM}"
    echo "${REV}-t${NORM}  --Define the cluster-based directory for tensorboard logging. Default is ${BOLD}${TENSORBOARD_PATH_CLUSTER}${NORM}"
    echo "${REV}-T${NORM}  --Define the image-based directory for tensorboard logging. Default is ${BOLD}${TENSORBOARD_PATH_IMAGE}${NORM}"
    echo "${REV}-n${NORM}  --Define the cluster-based directory for networks saving. Default is ${BOLD}${NETWORK_PATH_CLUSTER}${NORM}"
    echo "${REV}-N${NORM}  --Define the image-based directory for networks saving. Default is ${BOLD}${NETWORK_PATH_IMAGE}${NORM}"
    echo "${REV}-w${NORM}  --Define the cluster-based work directory. Default is ${BOLD}${CLUSTER_WORKDIR}${NORM}"
    echo "${REV}-W${NORM}  --Define the image-based work directory. Default is ${BOLD}${IMAGE_WORKDIR}${NORM}"
    echo "${REV}-s${NORM}  --Define the subdirectory for saving networks and tensorboards logs. Will be overwritten if the corresponding variable is in the config env file. Default is ${BOLD}${SAVE_INFO_PATH}${NORM}"
    echo "${REV}-p${NORM}  --Define the full image path to the executable program. Default is ${BOLD}${PROGRAM_PATH}${NORM}"
    echo -e "${REV}-h${NORM}  --Displays this help message"
    echo -e "Example: ${SCRIPT} -c conf.env -C /home/user/workdir/conf.env -i qec-mp.sif -t runs -T tmp_runs -n networks -N tmp_networks -w surface-rl-decoder -W /surface-rl-decoder -s test"\\n
    exit 1
}

### Start getopts code ###
while getopts d:c:C:i:I:t:T:n:N:s:w:W:p:h FLAG; do
    case $FLAG in
        h)
            HELP
            ;;
        d)
            JOB_DESCRIPTION=$OPTARG
            echo "-d used: $OPTARG"
            ;;
        c)
            CONFIG_FILE_NAME=$OPTARG
            SET_CONFIG_FILE_NAME=1
            echo "-c used: $OPTARG"
            ;;
        C)
            CONFIG_FILE=$OPTARG
            SET_FULL_CONFIG_FILE_PATH=1
            echo "-c used: $OPTARG"
            ;;
        i)
            IMAGE_NAME=$OPTARG
            echo "-i used: $OPTARG"
            ;;
        I)
            PATH_TO_IMAGE=$OPTARG
            SET_FULL_IMAGE_PATH=1
            echo "-I used: $OPTARG"
            ;;
        t)
            TENSORBOARD_PATH_CLUSTER=$OPTARG  
            echo "-t used: $OPTARG"
            ;;
        T)
            TENSORBOARD_PATH_IMAGE=$OPTARG  
            echo "-T used: $OPTARG"
            ;;
        n)
            NETWORK_PATH_CLUSTER=$OPTARG
            echo "-n used: $OPTARG"
            ;;
        N)
            NETWORK_PATH_IMAGE=$OPTARG
            echo "-N used: $OPTARG"
            ;;
        w)
            CLUSTER_WORKDIR=$OPTARG
            echo "-w used: $OPTARG"
            ;;
        W)
            IMAGE_WORKDIR=$OPTARG
            echo "-w used: $OPTARG"
            ;;
        s)
            OVERRIDE_SAVE_INFO_PATH=$OPTARG
            echo "-s used: $OPTARG"
            ;;
        p)
            PROGRAM_PATH=$OPTARG
            echo "-p used: $OPTARG"
            ;;
    esac
done

shift $((OPTIND-1))  #This tells getopts to move on to the next argument.

### Post processing of some variables
if [ -z "${JOB_DESCRIPTION}" ]; then
    echo "Error! Job description is required. Pass it as a string argument after the -d flag."
    exit 2
fi

if [ -z "$SET_FULL_IMAGE_PATH" ]; then
    PATH_TO_IMAGE=${CLUSTER_WORKDIR}/${IMAGE_NAME}
else
    PATH_TO_IMAGE=${PATH_TO_IMAGE}
fi

# option to override the save path for e.g. testing/developing
if [ -z "${OVERRIDE_SAVE_INFO_PATH}" ]; then
    SAVE_INFO_PATH=${SAVE_INFO_PATH}
else
    SAVE_INFO_PATH=${OVERRIDE_SAVE_INFO_PATH}
fi

# check if full path to config file is provided
if [ -z "$SET_FULL_CONFIG_FILE_PATH" ]; then
    # if not, check if a name for the config file was given
    if [ -z "$SET_CONFIG_FILE_NAME" ]; then
        CONFIG_FILE=
    # if it was, create the full path as .../workdir/conf-file
    else
        CONFIG_FILE=${CLUSTER_WORKDIR}/${CONFIG_FILE_NAME}
    fi
# if the full config path was given, use the full path with highest priority
else
    CONFIG_FILE=${CONFIG_FILE}
fi

ABS_TENSORBOARD_PATH_CLUSTER=${CLUSTER_WORKDIR}/${TENSORBOARD_PATH_CLUSTER}
ABS_NETWORK_PATH_CLUSTER=${CLUSTER_WORKDIR}/${NETWORK_PATH_CLUSTER}
ABS_TENSORBOARD_PATH_IMAGE=${IMAGE_WORKDIR}/${TENSORBOARD_PATH_IMAGE}
ABS_NETWORK_PATH_IMAGE=${IMAGE_WORKDIR}/${NETWORK_PATH_IMAGE}

if [ -z ${CONFIG_FILE} ]; then
    echo ""
    echo "No argument for env-config file supplied!"
    echo "Resort to default values"
    echo ""
    echo "Run image ${PATH_TO_IMAGE}, and execute the following:"
    echo ""

    set -x
    singularity run --nv \
      -B ${ABS_NETWORK_PATH_CLUSTER}:${ABS_NETWORK_PATH_IMAGE}:rw \
      -B ${ABS_TENSORBOARD_PATH_CLUSTER}:${ABS_TENSORBOARD_PATH_IMAGE}:rw \
      -B ${CLUSTER_WORKDIR}/custom_config:${IMAGE_WORKDIR}/custom_config:rw \
      --env CONFIG_GENERAL_SUMMARY_DATE="${SAVE_INFO_PATH}" \
      --env CONFIG_GENERAL_DESCRIPTION="${JOB_DESCRIPTION}" \
      ${PATH_TO_IMAGE} \
      /bin/bash -c \
        "cd ${IMAGE_WORKDIR}; \
        echo 'CONFIG_GENERAL_SUMMARY_DATE = ${CONFIG_GENERAL_SUMMARY_DATE}'; \
        python --version; \
        nvidia-smi; \
        python -c 'import torch; print(torch.cuda.is_available())'; \
        python -c 'import torch.cuda as tc; id = tc.current_device(); print(tc.get_device_name(id))';
        python ${IMAGE_WORKDIR}/${PROGRAM_PATH}"

    
else
    echo ""
    echo "Use config file ${CONFIG_FILE}"
    echo ""
    echo ""
    echo "Run image ${PATH_TO_IMAGE}, and execute the following:"
    echo ""

    set -x

    singularity run --nv \
      -B ${ABS_TENSORBOARD_PATH_CLUSTER}:${ABS_TENSORBOARD_PATH_IMAGE}:rw \
      -B ${ABS_NETWORK_PATH_CLUSTER}:${ABS_NETWORK_PATH_IMAGE}:rw \
      -B ${CLUSTER_WORKDIR}/custom_config:${IMAGE_WORKDIR}/custom_config:rw \
      --env CONFIG_GENERAL_SUMMARY_DATE="${SAVE_INFO_PATH}" \
      --env CONFIG_GENERAL_DESCRIPTION="${JOB_DESCRIPTION}" \
      --env-file ${CONFIG_FILE} \
      ${PATH_TO_IMAGE} \
      /bin/bash -c \
        "cd ${IMAGE_WORKDIR}; \
        echo 'CONFIG_GENERAL_SUMMARY_DATE = ${CONFIG_GENERAL_SUMMARY_DATE}'; \
        python --version; \
        nvidia-smi; \
        python -c 'import torch; print(torch.cuda.is_available())'; \
        python -c 'import torch.cuda as tc; id = tc.current_device(); print(tc.get_device_name(id))'; \
        python ${IMAGE_WORKDIR}/${PROGRAM_PATH}"
fi

set +x

#!/bin/bash
​
if [ -z "$1" ]
    then
    REGISTRY_USER="xero32"
else
    REGISTRY_USER="$1"
fi
​
if [ -z "$2" ]
    then
    IMAGE_NAME="qec-mp"
else
    IMAGE_NAME="$2"
fi
​
if [ -z "$3" ]
    then
    IMAGE_TAG=$(git symbolic-ref --short -q HEAD)
    IMAGE_TAG=$(echo "$IMAGE_TAG" | sed "s/\//-/g")
else
    IMAGE_TAG="$3"
    IMAGE_TAG=$(echo "$IMAGE_TAG" | sed "s/\//-/g")
fi
​
CUDA_TAG=cuda
​
docker build -t ${REGISTRY_USER}/${IMAGE_NAME}:${IMAGE_TAG} . 
docker push ${REGISTRY_USER}/${IMAGE_NAME}:${IMAGE_TAG}
​
singularity build -F ${IMAGE_NAME}_${IMAGE_TAG}.sif docker://${REGISTRY_USER}/${IMAGE_NAME}:${IMAGE_TAG}
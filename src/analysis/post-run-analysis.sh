#!/bin/bash
#SBATCH -J qec-eval
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:40:00
#SBATCH -A SNIC2020-33-2 -p alvis
#SBATCH --gpus-per-node=T4:1
#SBATCH --output=./logs-sbatch/logs-%j.out

# Job script to start the post-run evaluation routine on Alvis

set -x

singularity run --nv \
    -B /cephyr/NOBACKUP/groups/snic2021-23-319:/cephyr/NOBACKUP/groups/snic2021-23-319:rw \
    surface-rl-decoder/qec-mp_eval.sif \
    /bin/bash -c \
        "cd /surface-rl-decoder; \
        python --version; \
        nvidia-smi; \
        python -c 'import torch; print(torch.cuda.is_available())'; \
        python -c 'import torch.cuda as tc; id = tc.current_device(); print(tc.get_device_name(id))'; \
        python /surface-rl-decoder/src/analysis/analyze_threshold.py --run_evaluation --runs_config /cephyr/NOBACKUP/groups/snic2021-23-319/analysis/quick_runs_config.json --eval_job_id ${SLURM_JOB_ID} --max_steps=50 --network_path=/surface-rl-decoder/src/trained_models --p_stop=0.024 --p_start=0.0001 --p_step=0.001 --n_episodes=256 --max_recursion=0 --max_steps=60"

set +x

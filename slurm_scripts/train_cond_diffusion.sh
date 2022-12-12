#!/bin/bash
#SBATCH --partition=move --account=move --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=move1

#SBATCH --gres=gpu:1

#SBATCH --job-name="ready_7"
#SBATCH --output=/viscam/u/jiamanli/github/egoego_private/ready_diffusion_slurm_logs/slurm_output_train_ready_cond_motion_diffusion_amass_set7.log

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

source ~/.bashrc
export PATH="/viscam/u/jiamanli/my_gcc/gcc-5.4.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-11.3"
cd /viscam/u/jiamanli/github/egoego_private
conda activate goal

python trainer_amass_motion_diffusion_v2.py \
--window=120 \
--batch_size=32 \
--project="/viscam/projects/egoego/egoego_diffusion_output/ready_motion_diffusion_amass_runs/train" \
--exp_name="ready_cond_motion_diffusion_amass_set7" \
--wandb_pj_name="ready_cond_motion_diffusion_amass" \
--train_w_cond_diffusion \
--use_head_condition \
--use_first_pose_condition \
--add_noisy_head_condition \
--add_velocity_rep \
--use_tcn 
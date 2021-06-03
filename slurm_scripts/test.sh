#!/bin/bash

# Parameters
#SBATCH --constraint=volta32gb
#SBATCH --cpus-per-task=10
#SBATCH --error=/checkpoint/%u/jobs/%j.err
#SBATCH --gres=gpu:8
#SBATCH --job-name=vtf_test
#SBATCH --mem=450GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/checkpoint/%u/jobs/%j.out
#SBATCH --partition=learnfair
#SBATCH --signal=USR1@600
#SBATCH --comment=icml21-deadline
#SBATCH --time=12:00:00
# #SBATCH --mail-user=mandelapatrick@fb.com
# #SBATCH --mail-type=END,FAIL,REQUEUE

module load anaconda3
source activate pysf23_18

export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=19500

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000
echo $dist_url

if [ -z "$1" ]
then
	CFG='configs/Kinetics/ViT_base_ST_8x16.yaml'
else
	CFG=$1
fi
if [ -z "$2" ]
then
	CKPT_PATH='/checkpoint/mandelapatrick/slowfast/36328386/checkpoints/checkpoint_epoch_00030.pyth'
else
	CKPT_PATH=$2
fi


SAV_FOLDER="/checkpoint/${USER}/slowfast/${SLURM_JOB_ID}_test"
mkdir -p ${SAV_FOLDER}

# command
python tools/run_net.py --cfg $CFG \
NUM_GPUS 8 \
TRAIN.ENABLE False \
TEST.CHECKPOINT_FILE_PATH $CKPT_PATH  \
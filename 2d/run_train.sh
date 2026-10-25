#!/bin/bash
#
# 2D SCF-Net: data generation and training. Run it in the 2d directory.
#
# - One network learns the three input modes: image and seismicity (50% of the training samples), image only (30%,
#   --p_image) and seismicity only (20%, --p_meq).
# - The data generation runs in $nproc MPI processes (it needs mpi4py; without MPI, run several processes with
#   --ibeg and --iend instead); a 256 x 256 model takes 1.3 MB of disk space.
# - The training uses $gpus GPUs on each of $nodes nodes; the batch size is per GPU, and batch normalization uses the
#   batches of all GPUs. Without SLURM, Lightning starts one process per GPU. With SLURM, start one task per GPU, also
#   on one node: train="srun --nodes=$nodes --ntasks-per-node=$gpus python".
# - A new training in an existing $result directory overwrites its checkpoints.
#
set -e
[ -f generate_data.py ] || { echo "Run this script in the 2d directory"; exit 1; }

ntrain=2000                     # training samples
nvalid=200                      # validation samples
nproc=8                         # MPI processes of the data generation
mpirun="mpirun -np $nproc"      # MPI launcher, e.g., "srun -n $nproc" on a cluster
gpus=1                          # GPUs per node
nodes=1                         # nodes
train="python"                  # with SLURM: "srun --nodes=$nodes --ntasks-per-node=$gpus python"
result=result                   # checkpoints and logs
finetune=0                      # 1: also fine tune the network for the Opunake image (see below)

# Data generation: 256 x 256 models
OMP_NUM_THREADS=1 $mpirun python generate_data.py --mpi --outdir=dataset --ntrain=$ntrain --nvalid=$nvalid

# Training
$train ../src/main2.py \
	--ntrain=$ntrain --nvalid=$nvalid \
	--dir_data_train=dataset/data_train --dir_target_train=dataset/target_train \
	--dir_data_valid=dataset/data_valid --dir_target_valid=dataset/target_valid \
	--batch_train=8 --batch_valid=8 \
	--lr=2e-4 --epochs=60 --warmup=5 --bn_freeze=6 \
	--precision=bf16-mixed --gpus_per_node=$gpus --nodes=$nodes \
	--dir_output=$result

# Fine tuning for the Opunake image (as SCF-Net-FineTune of the paper): 256 x 1024 models that mimic the image and its
# seismicity, a smaller learning rate and fewer epochs
if [ "$finetune" = 1 ]; then
	OMP_NUM_THREADS=1 $mpirun python generate_data.py --mpi --preset=opunake --outdir=dataset_opunake \
		--ntrain=300 --nvalid=30 --n2=1024 --seed=7070
	$train ../src/main2.py \
		--finetune=$result/last.ckpt \
		--n1=256 --n2=1024 --ntrain=300 --nvalid=30 \
		--dir_data_train=dataset_opunake/data_train --dir_target_train=dataset_opunake/target_train \
		--dir_data_valid=dataset_opunake/data_valid --dir_target_valid=dataset_opunake/target_valid \
		--batch_train=4 --batch_valid=2 \
		--lr=2e-5 --epochs=20 --warmup=2 --bn_freeze=4 \
		--precision=bf16-mixed --gpus_per_node=$gpus --nodes=$nodes \
		--dir_output=${result}_opunake
fi

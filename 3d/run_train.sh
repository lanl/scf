#!/bin/bash
#
# 3D SCF-Net: data generation and training. Run it in the 3d directory.
#
# - One network learns the three input modes: image and seismicity (50% of the training samples), image only (30%,
#   --p_image) and seismicity only (20%, --p_meq).
# - The data generation runs in $nproc MPI processes (it needs mpi4py; without MPI, run several processes with
#   --ibeg and --iend instead); a 128 x 128 x 128 model takes about 5 s and 50 MB of disk space.
# - The training uses $gpus GPUs on each of $nodes nodes; the batch size is per GPU, and batch normalization uses the
#   batches of all GPUs. Without SLURM, Lightning starts one process per GPU. With SLURM, start one task per GPU, also
#   on one node: train="srun --nodes=$nodes --ntasks-per-node=$gpus python".
# - Every training sample is a random crop of 64 to $crop samples along each axis. With a batch size of 2, crops of up
#   to 96 samples need about 16 GB of GPU memory, and crops of up to 128 samples about 37 GB (estimated).
# - A new training in an existing $result directory overwrites its checkpoints.
#
set -e
[ -f generate_data.py ] || { echo "Run this script in the 3d directory"; exit 1; }

ntrain=1000                     # training samples
nvalid=100                      # validation samples
nproc=32                        # MPI processes of the data generation
mpirun="mpirun -np $nproc"      # MPI launcher, e.g., "srun -n $nproc" on a cluster
gpus=1                          # GPUs per node
nodes=1                         # nodes
train="python"                  # with SLURM: "srun --nodes=$nodes --ntasks-per-node=$gpus python"
crop=128                        # largest training crop (96 for GPUs with 24 GB)
result=result                   # checkpoints and logs

# Data generation: 128 x 128 x 128 models
OMP_NUM_THREADS=1 $mpirun python generate_data.py --mpi --outdir=dataset --ntrain=$ntrain --nvalid=$nvalid

# Training
$train ../src/main3.py \
	--ntrain=$ntrain --nvalid=$nvalid \
	--dir_data_train=dataset/data_train --dir_target_train=dataset/target_train \
	--dir_data_valid=dataset/data_valid --dir_target_valid=dataset/target_valid \
	--batch_train=2 --batch_valid=2 --crop_max=$crop \
	--lr=2e-4 --epochs=60 --warmup=5 --bn_freeze=6 \
	--precision=bf16-mixed --gpus_per_node=$gpus --nodes=$nodes \
	--dir_output=$result

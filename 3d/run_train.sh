
# ===============================================================================
# Generate data

make clean
make

# Data generation for 3D needs more computational resources; adjust omp threads and
# mpi ranks as needed.

export OMP_NUM_THREADS=16
mpirun -np 32 ./generate_data ntrain=6000 nvalid=600

# ===============================================================================
# Train models

# - The training uses 2 nodes and a total of 8 GPUs, so the equivalent batch size is 8
# - The base learning rate is 1.0e-4

for net in mtl scf scf-elastic
do
    python ../src/main3.py \
        --ntrain=6000 --nvalid=600 \
        --n1=128 --n2=128 --n3=128 \
        --batch_train=1 --batch_valid=1 --lr=1.0e-4 \
        --gpus_per_node=4 --nodes=2 \
        --net=$net \
        --dir_output=result_$net \
        --pp=y --ps=y --sp=n --ss=n \
        # To resume training from some checkpoint, use the following:
        # --resume=last
done

# ===============================================================================
# Validate models

# - The training uses 1 GPU

for net in mtl scf scf-elastic
do
    python ../src/main3.py \
        --ntrain=10 --nvalid=100 \
        --n1=128 --n2=128 --n3=128 \
        --batch_train=1 --batch_valid=1 \
        --gpus_per_node=1 --nodes=1 \
        --net=$net \
        --dir_output=result_$net \
        --pp=y --ps=y --sp=n --ss=n \
        --check=result_$net/last.ckpt
done

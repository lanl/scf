
# ===============================================================================
# Generate data

make clean
make

mpirun -np 10 ./generate_data ntrain=6000 nvalid=600 outdir=dataset
mpirun -np 10 ./generate_data_finetune ntrain=200 nvalid=20 outdir=dataset_finetune

# ===============================================================================
# Training

# - The following script uses 2 nodes and a total of 8 GPUs, therefore the equivalent batch size is 8
# - pp=y, ps=y, sp=n, ss=n only works for scf-elastic and will be ignored for other nets

for net in mtl mtl-seismicity scf scf-elastic
do
	python ../src/main2.py \
		--ntrain=6000 --nvalid=600 \
		--n1=256 --n2=256 \
		--batch_train=1 --batch_valid=1 --lr=1e-4 \
		--gpus_per_node=4 --nodes=2 \
		--epochs=100 \
		--net=$net \
		--pp=y --ps=y --sp=n --ss=n \
		--dir_output=result_$net
done

# ===============================================================================
# Validation

# - The following script uses 1 GPU

for net in mtl mtl-seismicity scf scf-elastic
do
	python ../src/main2.py \
		--ntrain=6000 --nvalid=600 \
		--n1=256 --n2=256 \
		--batch_train=1 --batch_valid=1 \
		--gpus_per_node=1 --nodes=1 \
		--net=$net \
		--pp=y --ps=y --sp=n --ss=n \
		--dir_output=result_$net \
		--check=result_$net/last.ckpt
done

# ===============================================================================
# Fine tune for Opunake

# - The following script uses 1 node and a total of 4 GPUs, therefore the equivalent batch size is 4
# - The base learning rate is 1.0e-5 rather than 1.0e-4 for this fine tuning
# - The maximum number of epochs is 50 rather than 100 for this fine tuning

net=scf
python ../src/main2.py \
    --net=$net \
    --ntrain=200 --nvalid=20 \
    --n1=256 --n2=1024 --epochs=50 --warmup=10 \
    --batch_train=1 --batch_valid=1 --lr=1e-5 \
    --gpus_per_node=4 --nodes=1 \
    --dir_data_train=dataset_finetune/data_train \
    --dir_target_train=dataset_finetune/target_train \
    --dir_data_valid=dataset_finetune/data_valid \
    --dir_target_valid=dataset_finetune/target_valid \
    --dir_output=result_$net''_finetune \
    --finetune=result_$net/last.ckpt


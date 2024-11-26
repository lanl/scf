
# Generate data
make clean
make
mpirun -np 10 ./exec_augmentation
mpirun -np 10 ./exec_finetune

# Train models
for dir in meq 
#nomeq meq meq_elastic
do
	python ./src/$dir/main2_infer.py \
		--ntrain=6000 --nvalid=600 \
		--n1=256 --n2=256 \
		--batch_train=8 --batch_valid=8 --lr=1e-4 \
		--gpus_per_node=1 --nodes=1 \
		--dir_output=result_$dir
done

# Validate models
for dir in meq
#nomeq meq meq_elastic
do
	python ./src/$dir/main2_infer.py \
		--ntrain=6000 --nvalid=600 \
		--n1=256 --n2=256 \
		--dir_data_valid=./dataset/data_valid \
		--dir_target_valid=./dataset/target_valid \
		--batch_train=1 --batch_valid=1 --lr=1e-4 \
		--gpus_per_node=1 --nodes=1 \
		--dir_output=result_$dir --check=result_$dir/last.ckpt
	exit
done

# Fine-tune
dir=meq
python ./src/$dir/main2_infer.py \
    --ntrain=200 --nvalid=20 \
    --n1=256 --n2=1024 --epochs=100 \
    --batch_train=1 --batch_valid=1 --lr=1e-4 \
    --gpus_per_node=1 --nodes=1 \
    --dir_data_train=dataset_finetune/data_train \
    --dir_target_train=dataset_finetune/target_train \
    --dir_data_valid=dataset_finetune/data_valid \
    --dir_target_valid=dataset_finetune/target_valid \
    --dir_output=result_$dir''_finetune \
    --finetune=result_$dir/last.ckpt


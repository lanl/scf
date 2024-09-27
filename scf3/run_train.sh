
# Generate data
make clean
make
mpirun -np 10 ./exec

# Train models
# For validation, set --check=result_$dir/last.ckpt or epoch=xx.ckpt
for dir in nomeq meq meq_elastic
do

	python ./src/$dir/main3_infer.py \
		    --ntrain=2000 --nvalid=200 \
		    --n1=128 --n2=256 --n3=256 \
		    --batch_train=1 --batch_valid=1 --lr=0.5e-4 \
		    --gpus_per_node=4 --nodes=2 \
		    --dir_output=result_$dir

done

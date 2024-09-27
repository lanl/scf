
# Generate data
make clean
make
mpirun -np 10 ./exec

# Train models
# For validation, set --check=result_$dir/last.ckpt or epoch=xx.ckpt
for dir in nomeq meq meq_elastic
do
	python ./src/$dir/main2_infer.py \
		--ntrain=6000 --nvalid=600 \
		--n1=256 --n2=256 \
		--batch_train=8 --batch_valid=8 --lr=1e-4 \
		--gpus_per_node=1 --nodes=1 \
		--dir_output=result_$dir
done


dir=meq
python ./src/$dir/main3_infer.py \
    --ntrain=2000 --nvalid=200 \
    --n1=128 --n2=128 --n3=128 \
    --batch_train=1 --batch_valid=1 --lr=1e-4 \
    --gpus_per_node=4 --nodes=1 \
    --dir_output=result_$dir''

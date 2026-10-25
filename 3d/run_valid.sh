#!/bin/bash
#
# 3D SCF-Net: validation of a trained network for the three input modes (image only, image and seismicity, seismicity
# only). Run it in the 3d directory after run_train.sh. Results in $out:
#
#   metrics_validation.txt, .json, .png   precision, recall, F1, and dip and strike errors of every mode on the
#                                         validation set, and the agreement of the faults from both inputs with those
#                                         from the image only
#   predict_<i>.png, predict_<i>.<mode>.fsem
#                                         true and predicted faults of the first validation volumes
#   blocks.json                           block-wise against whole-volume inference of every mode
#   f3.png, f3.<mode>.fsem/.fdip/.fstrike faults of the F3 volume (North Sea, Netherlands) from every mode, block by block
#
set -e
[ -f generate_data.py ] || { echo "Run this script in the 3d directory"; exit 1; }

nvalid=100                      # validation samples
model=result/last.ckpt          # trained network
out=result/validation           # results
mkdir -p $out

# Validation set
python ../src/main3.py --check=$model --nvalid=$nvalid \
	--dir_data_valid=dataset/data_valid --dir_target_valid=dataset/target_valid \
	--dir_output=$out

# Block-wise against whole-volume inference, on 128 x 192 x 192 volumes (the whole volume must fit in the GPU memory)
[ -d dataset_large ] || python generate_data.py --outdir=dataset_large --ntrain=0 --nvalid=4 --n2=192 --n3=192 --nf 3 18 --seed=777
python evaluate.py --dataset=dataset_large --nsample=4 --n1=128 --n2=192 --n3=192 --blocks=64:32,96:32 \
	--model=$model --out=$out/blocks.json

# F3 volume of 128 x 256 x 256 samples, block by block; test/f3_img.zip is split in parts, which unzip extracts after
# they are joined (with a warning about the joined parts)
if [ ! -f test/dutch.img ]; then
	cat test/f3_img.z01 test/f3_img.z02 test/f3_img.z03 test/f3_img.zip > test/f3_img_joined.zip
	unzip -o -q test/f3_img_joined.zip -d test || true
	rm -f test/f3_img_joined.zip
fi
[ -f test/dutch.meq ] || unzip -o -q test/f3_meq.zip -d test
[ "$(stat -c %s test/dutch.img)" = 33554432 ] || { echo "test/dutch.img is not 128 x 256 x 256 floats"; exit 1; }
python ../src/main3.py --model=$model --input=test/dutch.img --input_meq=test/dutch.meq \
	--n1=128 --n2=256 --n3=256 --mode=all --b1=128 --b2=128 --b3=128 --p1=32 --p2=32 --p3=32 --output=$out/f3
python plot_results.py --image=test/dutch.img --meq=test/dutch.meq --prefix=$out/f3 \
	--n1=128 --n2=256 --n3=256 --out=$out/f3.png

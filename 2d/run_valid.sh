#!/bin/bash
#
# 2D SCF-Net: validation of a trained network for the three input modes (image only, image and seismicity, seismicity
# only). Run it in the 2d directory after run_train.sh. Results in $out:
#
#   metrics_validation.txt, .json, .png   precision, recall, F1 and dip error of every mode on the validation set, and
#                                         the agreement of the faults from both inputs with those from the image only
#   predict_<i>.png, predict_<i>.<mode>.fsem
#                                         true and predicted faults of the first validation images
#   blocks.json                           block-wise against whole-image inference of every mode
#   opunake.png, opunake.<mode>.fsem/.fdip
#                                         faults of the Opunake image from every mode, block by block
#
set -e
[ -f generate_data.py ] || { echo "Run this script in the 2d directory"; exit 1; }

nvalid=200                      # validation samples
model=result/last.ckpt          # trained network (result_opunake/last.ckpt for the fine-tuned network)
out=result/validation           # results
mkdir -p $out

# Validation set
python ../src/main2.py --check=$model --nvalid=$nvalid \
	--dir_data_valid=dataset/data_valid --dir_target_valid=dataset/target_valid \
	--dir_output=$out

# Block-wise against whole-image inference, on 256 x 1024 images
python generate_data.py --outdir=dataset_wide --ntrain=0 --nvalid=10 --n2=1024 --nf 4 40 --seed=777
python evaluate.py --dataset=dataset_wide --nsample=10 --n1=256 --n2=1024 --blocks=128:32,128:64,256:64 \
	--model=$model --out=$out/blocks.json

# Opunake image, block by block
python ../src/main2.py --model=$model --input=test/opunake.bin --input_meq=test/opunake.bin.meq \
	--n1=256 --n2=1024 --mode=all --b1=256 --b2=256 --p1=64 --p2=64 --output=$out/opunake
python plot_results.py --image=test/opunake.bin --meq=test/opunake.bin.meq --prefix=$out/opunake \
	--n1=256 --n2=1024 --out=$out/opunake.png

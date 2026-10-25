#
# Block-wise inference against whole-volume inference of 3D SCF-Net, on the validation volumes of a data set, for the
# three input modes: the mean absolute difference of the fault probability, and the F1 of the block-wise faults
# against the whole-volume faults (distance tolerance --tol of main3.py). The whole volume must fit in the GPU memory.
#
# Usage: python evaluate.py --dataset=<dir> --nsample=<n> --blocks=64:32,96:32 --out=<file.json> --model=<checkpoint>
#        --n1=<n1> --n2=<n2> --n3=<n3> (and other options of src/main3.py)
#

import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser(description='Block-wise against whole-volume inference of 3D SCF-Net')
ap.add_argument('--dataset', type=str, required=True, help='data set directory with data_valid')
ap.add_argument('--nsample', type=int, default=4, help='number of validation volumes')
ap.add_argument('--blocks', type=str, default='64:32,96:32', help='block sizes and overlaps, size:overlap separated by commas')
ap.add_argument('--out', type=str, required=True, help='output file (.json)')
args, rest = ap.parse_known_args()

sys.argv = [sys.argv[0]] + rest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch
from main3 import opts, MODES, load_model, normalize_image, predict
from utility import read_array
from scf_common import fault_scores

net = load_model(opts.model)
shape = (1, 1, opts.n1, opts.n2, opts.n3)
configs = [tuple(int(v) for v in c.split(':')) for c in args.blocks.split(',')]

res = {}
for i in range(args.nsample):
    d = '%s/data_valid/%d' % (args.dataset, i)
    img = normalize_image(read_array(d + '_img.bin', shape))
    meq = read_array(d + '_meq.bin', shape)
    for mode in MODES:
        x = {'img': torch.zeros_like(img) if mode == 'meq' else img, 'meq': torch.zeros_like(meq) if mode == 'image' else meq}
        whole = predict(net, x)['fsem']
        for b, o in configs:
            blk = predict(net, x, block=(b, b, b), overlap=(o, o, o))['fsem']
            res.setdefault((mode, b, o), []).append([float(torch.mean(torch.abs(blk - whole))),
                                                     float(fault_scores(blk, whole, tol=opts.tol)[2])])
    print('volume %d done' % i, flush=True)

summary = {'%s %d:%d' % k: {'mean_abs_diff': float(np.mean([v[0] for v in r])), 'f1_vs_whole': float(np.mean([v[1] for v in r]))}
           for k, r in res.items()}
with open(args.out, 'w') as f:
    json.dump(summary, f, indent=1)
for k, v in summary.items():
    mode, blk = k.split(' ')
    print('%-6s block:overlap %-7s mean |dp| %.2e  F1 against the whole volume %.4f' % (mode, blk, v['mean_abs_diff'], v['f1_vs_whole']))

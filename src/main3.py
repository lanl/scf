#
# (c) 2025-2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by
# Triad National Security, LLC for the U.S. Department of Energy/National
# Nuclear Security Administration. All rights in the program are reserved
# by Triad National Security, LLC, and the U.S. Department of Energy/
# National Nuclear Security Administration.
# The Government is granted for itself and others acting on its behalf a nonexclusive,
# paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly
# and display publicly, and to permit others to do so.
#
# Author:
#   Kai Gao, kaigao@lanl.gov
#
# SCF-Net in 3D: one network detects faults, and their dip and strike, from a seismic image and a seismicity
# location image, from the image only, or from the seismicity only.
#   training:   python main3.py [options]
#   validation: python main3.py --check=<checkpoint> [options]
#   inference:  python main3.py --model=<checkpoint> --input=<image> --input_meq=<seismicity> --output=<prefix> [options]
#

import warnings
import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utility import read_array, write_array, set_random_seed, get_numpy, date_time, set_font
from scf_common import (estimate_scale, agc, apply_mode, CropCollate, random_subcrop, dice_loss, bce_loss, kl_loss,
                        masked_l1, dilate, fault_scores, predict_blocks)
from seismicity import read_events, render_seismicity_nd
from model3 import resu1, resu2, resu3, maxpool, LocalSpatialChannelAttention, mtl_decoder_fusion, mtl_subdecoder

# ==============================================================================
parser = argparse.ArgumentParser(description='SCF-Net 3D: training (default), validation (--check) and inference (--model)')
# Data
parser.add_argument('--n1', type=int, default=128, help='number of samples along x1 (depth)')
parser.add_argument('--n2', type=int, default=128, help='number of samples along x2')
parser.add_argument('--n3', type=int, default=128, help='number of samples along x3')
parser.add_argument('--ntrain', type=int, default=1000, help='number of training samples')
parser.add_argument('--nvalid', type=int, default=100, help='number of validation samples')
parser.add_argument('--dir_data_train', type=str, default='./dataset/data_train', help='directory of the training inputs')
parser.add_argument('--dir_target_train', type=str, default='./dataset/target_train', help='directory of the training labels')
parser.add_argument('--dir_data_valid', type=str, default='./dataset/data_valid', help='directory of the validation inputs')
parser.add_argument('--dir_target_valid', type=str, default='./dataset/target_valid', help='directory of the validation labels')
parser.add_argument('--dir_output', type=str, default='./result', help='directory of the checkpoints, logs and validation results')
# Training
parser.add_argument('--batch_train', type=int, default=2, help='training batch size per GPU')
parser.add_argument('--batch_valid', type=int, default=2, help='validation batch size per GPU')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--warmup', type=int, default=5, help='number of epochs of linear learning-rate warmup')
parser.add_argument('--lr', type=float, default=2.0e-4, help='learning rate after the warmup, with a cosine decay')
parser.add_argument('--bn_freeze', type=int, default=6, help='number of final epochs with fixed batch-normalization statistics')
parser.add_argument('--threads', type=int, default=8, help='number of data-loader workers per GPU')
parser.add_argument('--precision', type=str, default='32', help='32, bf16-mixed or 16-mixed')
parser.add_argument('--seed', type=int, default=12345, help='random seed')
parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume the training from')
parser.add_argument('--finetune', type=str, default=None, help='checkpoint that initializes the network (fine tuning)')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus_per_node', type=int, default=1,
                    help='number of GPUs per node (with --accelerator=cpu, the number of CPU processes)')
parser.add_argument('--accelerator', type=str, default='auto', choices=['auto', 'gpu', 'cpu'],
                    help='auto: the GPUs if there are any')
# Training samples
parser.add_argument('--crop1', type=int, default=64, help='smallest random crop along x1 (0: no crop)')
parser.add_argument('--crop2', type=int, default=64, help='smallest random crop along x2 (0: no crop)')
parser.add_argument('--crop3', type=int, default=64, help='smallest random crop along x3 (0: no crop)')
parser.add_argument('--crop_max', type=int, default=0,
                    help='largest random crop along every axis (0: the sample size); smaller crops need less GPU memory')
parser.add_argument('--p_image', type=float, default=0.3, help='probability of a training sample with the image only')
parser.add_argument('--p_meq', type=float, default=0.2, help='probability of a training sample with the seismicity only')
parser.add_argument('--gain', type=float, nargs=2, default=[0.5, 2.0], help='range of the random gain of the training images')
parser.add_argument('--flip', type=float, default=0.5, help='probability of a reversed polarity of a training image')
parser.add_argument('--meq_blur', type=float, nargs=2, default=[0.0, 1.5],
                    help='range of the standard deviation (voxels) of the random blur of the seismicity of half of the training samples')
# Loss
parser.add_argument('--w_dip', type=float, default=1.0, help='weight of the dip and strike losses')
parser.add_argument('--w_bce', type=float, default=1.0,
                    help='weight of the binary cross entropy of the fault probability, added to its Dice loss; a smaller weight '
                    'gives fewer low-probability pixels, and the patch-consistency loss needs a positive weight')
parser.add_argument('--w_consist', type=float, default=1.0,
                    help='weight of the patch-consistency loss; it needs the cross entropy (--w_bce > 0), without which it '
                    'drives the faults to a constant')
# Validation
parser.add_argument('--check', type=str, default=None, help='checkpoint to validate on the validation set')
parser.add_argument('--tol', type=int, default=2, help='distance tolerance of the fault metrics (voxels)')
parser.add_argument('--nplot', type=int, default=10, help='number of validation samples to plot and save')
# Inference
parser.add_argument('--model', type=str, default=None, help='checkpoint for inference')
parser.add_argument('--input', type=str, default=None, help='image (raw float32, x1 fastest); zero if not given')
parser.add_argument('--input_meq', type=str, default=None, help='seismicity location image (raw float32); zero if not given')
parser.add_argument('--input_events', type=str, default=None,
                    help='event locations instead of --input_meq: a text or CSV file with x1, x2 and x3 grid coordinates '
                    '(from 0) in its first columns')
parser.add_argument('--output', type=str, default=None, help='prefix of the outputs <output>[.<mode>].fsem, .fdip and .fstrike')
parser.add_argument('--fault_threshold', type=float, default=0.5,
                    help='fault probability above which the dip and strike outputs hold the predicted dip and strike; zero elsewhere')
parser.add_argument('--mode', type=str, default='all', choices=['all', 'both', 'image', 'meq'],
                    help='input mode: image and seismicity, image only, seismicity only, or all three')
parser.add_argument('--b1', type=int, default=0, help='block size along x1 (0: the whole axis)')
parser.add_argument('--b2', type=int, default=0, help='block size along x2 (0: the whole axis)')
parser.add_argument('--b3', type=int, default=0, help='block size along x3 (0: the whole axis)')
parser.add_argument('--p1', type=int, default=32, help='block overlap along x1')
parser.add_argument('--p2', type=int, default=32, help='block overlap along x2')
parser.add_argument('--p3', type=int, default=32, help='block overlap along x3')
parser.add_argument('--scale', type=str, default='rms', choices=['rms', 'agc'],
                    help='amplitude normalization of the input image: rms of the whole image, or agc (local rms)')
parser.add_argument('--agc_sigma', type=float, default=32.0, help='window of agc (voxels)')
opts = parser.parse_args()

assert min(opts.n1, opts.n2, opts.n3) >= 1
assert 0 <= opts.p_image + opts.p_meq <= 1
assert opts.w_bce >= 0
if opts.model is None and opts.check is None:
    assert opts.w_bce > 0 or opts.w_consist == 0, \
        'the patch-consistency loss (--w_consist) needs the cross entropy (--w_bce > 0): without it, the faults collapse to a constant'

use_gpu = opts.accelerator != 'cpu' and torch.cuda.is_available() and opts.gpus_per_node >= 1
if opts.accelerator == 'gpu' and not use_gpu:
    raise RuntimeError('--accelerator=gpu, but no GPU is available')
device = torch.device('cuda' if use_gpu else 'cpu')
torch.set_float32_matmul_precision('high')

# Input modes, with the weights of their validation losses
MODES = ['image', 'both', 'meq']
TITLES = {'image': 'Image Only', 'both': 'Image + Seismicity', 'meq': 'Seismicity Only'}
WEIGHTS = {'both': 1.0 - opts.p_image - opts.p_meq, 'image': opts.p_image, 'meq': opts.p_meq}


# ==============================================================================
class BasicDataset(Dataset):
    """Samples <i>_img.bin and <i>_meq.bin, and labels <i>_fsem.bin, _fdip.bin, _fstrike.bin and _fsem_meq.bin."""

    def __init__(self, dir_data, dir_target, ids):
        self.dir_data = dir_data
        self.dir_target = dir_target
        self.ids = ids
        self.shape = (1, opts.n1, opts.n2, opts.n3)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = self.dir_data + '/' + str(self.ids[i])
        t = self.dir_target + '/' + str(self.ids[i])
        img = read_array(d + '_img.bin', self.shape)
        data = {'img': img / estimate_scale(img.unsqueeze(0))[0], 'meq': read_array(d + '_meq.bin', self.shape)}
        target = {k: read_array(t + '_' + k + '.bin', self.shape) for k in ['fsem', 'fdip', 'fstrike']}
        # Faults indicated by the seismicity, the label of the seismicity-only mode; for data sets without it,
        # the faults within 4 voxels of a seismicity location probability above 0.1
        if os.path.isfile(t + '_fsem_meq.bin'):
            target['fsem_meq'] = read_array(t + '_fsem_meq.bin', self.shape)
        else:
            near = torch.nn.functional.max_pool3d(data['meq'].unsqueeze(0), 9, 1, 4)[0] > 0.1
            target['fsem_meq'] = target['fsem'] * near.float()
        return data, target


def custom_loss(y_pred, y_true):
    """Dice loss and binary cross entropy (weight --w_bce) of the fault probability, and L1 of the dip and strike on the true faults."""

    mt = y_true['fsem']
    loss_fsem = dice_loss(y_pred['fsem'], mt)
    if opts.w_bce > 0:
        loss_fsem = loss_fsem + opts.w_bce * bce_loss(y_pred['fsem'], mt)
    loss_fdip = masked_l1(y_pred['fdip_raw'], y_true['fdip'], mt) * opts.w_dip
    loss_fstrike = masked_l1(y_pred['fstrike_raw'], y_true['fstrike'], mt) * opts.w_dip

    return loss_fsem + loss_fdip + loss_fstrike, loss_fsem, loss_fdip, loss_fstrike


# ==============================================================================
class mtlnet(pl.LightningModule):

    def __init__(self, lr=2.0e-4, warmup_epochs=5, max_epochs=60):

        super(mtlnet, self).__init__()

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.l1 = 32
        self.l2 = 64
        self.l3 = 128

        # Image and seismicity encoders, fused at the bottleneck and in the decoder
        self.encoder1_img = resu1(1, self.l1)
        self.encoder2_img = resu2(self.l1, self.l2)
        self.encoder1_meq = resu1(1, self.l1)
        self.encoder2_meq = resu2(self.l1, self.l2)
        self.fusion3 = LocalSpatialChannelAttention((self.l2, self.l2), 2 * self.l2, channel_ratio=8)
        self.encoder3 = resu3(2 * self.l2, self.l3)
        self.decoder = mtl_decoder_fusion(self.l1, self.l2, self.l3, out_ch=self.l1, last_kernel_size=3)

        # Heads of the fault probability, dip / 180 and strike / 180
        self.subdecoder_fault_semantic = mtl_subdecoder(in_ch=self.l1, out_ch=1, bn=False, mid_activation='relu', activation='sigmoid')
        self.subdecoder_fault_dip = mtl_subdecoder(in_ch=self.l1, out_ch=1, bn=True, mid_activation='relu', activation='sigmoid')
        self.subdecoder_fault_strike = mtl_subdecoder(in_ch=self.l1, out_ch=1, bn=True, mid_activation='relu', activation='sigmoid')

    def forward(self, x):

        e1_img = self.encoder1_img(x['img'])
        e2_img = self.encoder2_img(maxpool(e1_img, 2))
        e1_meq = self.encoder1_meq(x['meq'])
        e2_meq = self.encoder2_meq(maxpool(e1_meq, 2))
        e3 = self.encoder3(self.fusion3(maxpool(e2_img, 2), maxpool(e2_meq, 2)))
        f = self.decoder(x['img'], e1_img, e2_img, e1_meq, e2_meq, e3)

        # The dip and the strike are learned on the faults only (fdip_raw, fstrike_raw), and the outputs are the raw dip and
        # strike where the fault probability exceeds the threshold, zero elsewhere
        out = {'fsem': self.subdecoder_fault_semantic(f), 'fdip_raw': self.subdecoder_fault_dip(f),
               'fstrike_raw': self.subdecoder_fault_strike(f)}
        on = (out['fsem'] > opts.fault_threshold).to(out['fdip_raw'].dtype)
        out['fdip'] = out['fdip_raw'] * on
        out['fstrike'] = out['fstrike_raw'] * on

        return out

    def freeze_bn(self):
        # Fixed batch-normalization statistics in the final epochs, so that the training ends with the
        # normalization of inference
        if opts.bn_freeze > 0 and self.current_epoch >= self.max_epochs - opts.bn_freeze:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def on_train_epoch_start(self):

        self.freeze_bn()

    def on_train_batch_start(self, batch, batch_idx):

        self.freeze_bn()

    def training_step(self, batch, batch_idx):

        x, y_true = batch
        y_pred = self.forward(x)

        loss, loss_fsem, loss_fdip, loss_fstrike = custom_loss(y_pred, y_true)
        self.log('train_loss_fsem', loss_fsem, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss_fdip', loss_fdip, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss_fstrike', loss_fstrike, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_f1', fault_scores(y_pred['fsem'].detach(), y_true['fsem'], tol=opts.tol)[2], on_step=False, on_epoch=True,
                 sync_dist=True)

        # Patch consistency: the faults of a random sub-crop should be those of the whole crop, including near the
        # sub-crop edges, so that block-wise inference reproduces whole-volume inference with small blocks
        if opts.w_consist > 0:
            sl = random_subcrop(x['img'].shape[2:])
            y_sub = self.forward({k: v[sl] for k, v in x.items()})
            loss_consist = kl_loss(y_sub['fsem'], y_pred['fsem'][sl].detach())
            self.log('train_loss_consist', loss_consist, on_step=False, on_epoch=True, sync_dist=True)
            loss = loss + opts.w_consist * loss_consist

        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        loss = 0.0
        for mode in MODES:
            xm, y_true = apply_mode(x, y, mode)
            y_pred = self.forward(xm)
            l = custom_loss(y_pred, y_true)[0]
            self.log('valid_loss_' + mode, l, on_step=False, on_epoch=True, sync_dist=True)
            self.log('valid_f1_' + mode, fault_scores(y_pred['fsem'], y_true['fsem'], tol=opts.tol)[2], on_step=False,
                     on_epoch=True, sync_dist=True)
            loss = loss + WEIGHTS[mode] * l
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.lr)

        def lr_lambda(epoch):
            # linear warmup and cosine decay
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': LambdaLR(optimizer, lr_lambda=lr_lambda), 'interval': 'epoch'}}


def load_model(path):

    net = mtlnet()
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['state_dict'])

    return net.to(device).eval()


def normalize_image(img):
    """Amplitude normalization of a whole image (1, 1, n1, n2, n3), before any block-wise inference."""

    if opts.scale == 'agc':
        return agc(img, opts.agc_sigma)
    return img / estimate_scale(img)


def predict(net, data, block=(0, 0, 0), overlap=(0, 0, 0)):
    """
    Fault probability, dip and strike of a whole volume (data: dictionary of (1, 1, n1, n2, n3) tensors), from the
    whole volume or block by block. The volume can stay in the CPU memory: every block is moved to the device of
    the network, and the outputs are blended on the device of the inputs.
    """

    def fn(d):
        with torch.no_grad():
            if opts.precision in ('16-mixed', 'bf16-mixed'):
                dtype = torch.float16 if opts.precision == '16-mixed' else torch.bfloat16
                with torch.autocast(device_type=device.type, dtype=dtype):
                    out = net(d)
            else:
                out = net(d)
        return {k: out[k].float() for k in ['fsem', 'fdip_raw', 'fstrike_raw']}

    out = predict_blocks(fn, data, block, overlap, device=device)
    on = (out['fsem'] > opts.fault_threshold).to(out['fdip_raw'].dtype)
    out['fdip'] = out['fdip_raw'] * on
    out['fstrike'] = out['fstrike_raw'] * on

    return out


# ==============================================================================
class Scores:
    """Fault precision, recall and F1 with a distance tolerance over a data set, and dip and strike errors."""

    def __init__(self):

        self.c = np.zeros(4)
        self.e = np.zeros(3)
        self.loss = 0.0
        self.n = 0

    def add(self, p, t, out=None, label=None, loss=None):

        pb = (p > 0.5).float()
        tb = (t > 0.5).float()
        self.c += [float(torch.sum(pb * dilate(tb, opts.tol))), float(torch.sum(pb)), float(torch.sum(tb * dilate(pb, opts.tol))),
                   float(torch.sum(tb))]
        if out is not None:
            # dip and strike errors (degrees) of the true faults found; the strike difference is taken modulo 180
            on = (pb * tb) > 0
            dd = torch.abs(out['fdip_raw'] - label['fdip'])[on] * 180.0
            ds = torch.abs(out['fstrike_raw'] - label['fstrike'])[on] * 180.0
            self.e += [float(torch.sum(dd)), float(torch.sum(torch.minimum(ds, 180.0 - ds))), float(torch.sum(on))]
        if loss is not None:
            self.loss += float(loss)
            self.n += 1

    def result(self):

        tp_p, n_p, tp_r, n_t = self.c
        p = tp_p / n_p if n_p > 0 else (1.0 if n_t == 0 else 0.0)
        r = tp_r / n_t if n_t > 0 else (1.0 if n_p == 0 else 0.0)
        out = {'precision': p, 'recall': r, 'f1': 2 * p * r / (p + r) if p + r > 0 else 0.0}
        if self.e[2] > 0:
            out['dip_error'] = self.e[0] / self.e[2]
            out['strike_error'] = self.e[1] / self.e[2]
        if self.n > 0:
            out['loss'] = self.loss / self.n
        return out


def plot_sample(img, meq, fsem, fsem_meq, pred, filename):
    """
    Faults of one validation volume on three orthogonal slices through its center: the true faults, the faults from
    the image only, from the image and the seismicity, the faults that the seismicity indicates, and the faults from
    the seismicity only. Faults (probability at least 0.5) are red; seismicity is blue on the image and black on
    white alone.
    """

    n1, n2, n3 = img.shape
    cuts = [(n1 // 2, slice(None), slice(None)), (slice(None), n2 // 2, slice(None)), (slice(None), slice(None), n3 // 2)]
    rows = [('True Faults', 'img', False, fsem), ('Image Only', 'img', False, pred['image']),
            ('Image + Seismicity', 'img', True, pred['both']), ('Seismicity Faults', 'meq', False, fsem_meq),
            ('Seismicity Only', 'meq', False, pred['meq'])]
    red = ListedColormap(['#ff0000'])
    blue = ListedColormap(['#0000ff'])
    fig, ax = plt.subplots(len(rows), 3, figsize=(7.2, 2.25 * len(rows)), gridspec_kw={'width_ratios': [n3, n3, n2]}, squeeze=False)
    for r, (title, back, with_meq, faults) in enumerate(rows):
        for c, sl in enumerate(cuts):
            a = ax[r][c]
            if back == 'meq':
                a.imshow(meq[sl], cmap='binary', vmin=0, vmax=1, interpolation='none', aspect='equal')
            else:
                a.imshow(img[sl], cmap='gray', vmin=-2.5, vmax=2.5, interpolation='none', aspect='equal')
                if with_meq:
                    a.imshow(np.ma.masked_where(meq[sl] < 0.5, meq[sl]), cmap=blue, interpolation='none', aspect='equal')
            a.imshow(np.ma.masked_where(faults[sl] < 0.5, faults[sl]), cmap=red, interpolation='none', aspect='equal')
            a.set_xticks([])
            a.set_yticks([])
            if r == 0:
                a.set_title(['X1 Slice', 'X2 Slice', 'X3 Slice'][c])
            if c == 0:
                a.set_ylabel(title)
    plt.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.3)
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def plot_scores(scores, filename):
    """Precision, recall and F1, and the dip and strike errors, of the three input modes."""

    colors = {'image': '#0000ff', 'both': '#ff0000', 'meq': '#000000'}
    w = 0.26
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={'width_ratios': [3, 2]})
    for j, mode in enumerate(MODES):
        s = scores[mode]
        ax[0].bar(np.arange(3) + (j - 1) * w, [s['precision'], s['recall'], s['f1']], w, color=colors[mode], label=TITLES[mode])
        ax[1].bar(np.arange(2) + (j - 1) * w, [s.get('dip_error', 0.0), s.get('strike_error', 0.0)], w, color=colors[mode])
    ax[0].set_xticks(np.arange(3), ['Precision', 'Recall', 'F1'])
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Score')
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    ax[1].set_xticks(np.arange(2), ['Dip', 'Strike'])
    ax[1].set_ylabel('Mean Error (Degrees)')
    plt.tight_layout(pad=0.3)
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


# ==============================================================================
if __name__ == '__main__':

    if opts.model is None and opts.check is None:
        ## Training

        set_random_seed(opts.seed)
        # data-loader workers are seeded by their rank too, so that every GPU draws its own crops, modes and gains
        pl.seed_everything(opts.seed, workers=True)

        # a new training in the same directory overwrites last.ckpt (instead of writing last-v1.ckpt)
        checkpoint = ModelCheckpoint(monitor='valid_loss', dirpath=opts.dir_output, filename='{epoch:d}', mode='min',
                                     save_top_k=3, save_last=True, enable_version_counter=False)
        params = {'max_epochs': opts.epochs, 'default_root_dir': opts.dir_output, 'logger': TensorBoardLogger(opts.dir_output),
                  'callbacks': [checkpoint, LearningRateMonitor(logging_interval='epoch')], 'precision': opts.precision,
                  'accelerator': 'gpu' if use_gpu else 'cpu', 'devices': max(1, opts.gpus_per_node), 'num_nodes': opts.nodes}
        if params['devices'] * opts.nodes > 1:
            # one process per GPU; batch normalization uses the batches of all GPUs
            params['strategy'] = 'ddp'
            params['sync_batchnorm'] = use_gpu
        trainer = pl.Trainer(**params)

        collate = CropCollate((opts.crop1, opts.crop2, opts.crop3), (opts.crop_max,) * 3, opts.p_image, opts.p_meq, opts.gain,
                              opts.flip, opts.meq_blur)
        train_loader = DataLoader(BasicDataset(opts.dir_data_train, opts.dir_target_train, np.arange(opts.ntrain)),
                                  batch_size=opts.batch_train, num_workers=opts.threads, shuffle=True, collate_fn=collate,
                                  drop_last=True, pin_memory=use_gpu, persistent_workers=opts.threads > 0)
        valid_loader = DataLoader(BasicDataset(opts.dir_data_valid, opts.dir_target_valid, np.arange(opts.nvalid)),
                                  batch_size=opts.batch_valid, num_workers=opts.threads, pin_memory=use_gpu,
                                  persistent_workers=opts.threads > 0)

        net = mtlnet(lr=opts.lr, warmup_epochs=opts.warmup, max_epochs=opts.epochs)
        if opts.finetune is not None:
            net.load_state_dict(torch.load(opts.finetune, map_location='cpu', weights_only=False)['state_dict'])

        trainer.fit(net, train_loader, valid_loader, ckpt_path=opts.resume)

        print(date_time(), ' >> Training finished')

    if opts.model is None and opts.check is not None:
        ## Validation: metrics of the three input modes, and figures and fault probabilities of the first samples

        set_font()
        plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9})
        os.makedirs(opts.dir_output, exist_ok=True)
        net = load_model(opts.check)
        valid_loader = DataLoader(BasicDataset(opts.dir_data_valid, opts.dir_target_valid, np.arange(opts.nvalid)), batch_size=1,
                                  num_workers=min(opts.threads, 4))

        scores = {m: Scores() for m in MODES + ['agreement']}
        for i, (data, label) in enumerate(tqdm(valid_loader, desc='validation', unit='volume')):
            pred = {}
            for mode in MODES:
                x, y = apply_mode(data, label, mode)
                x = {k: v.to(device) for k, v in x.items()}
                y = {k: v.to(device) for k, v in y.items()}
                out = predict(net, x)
                scores[mode].add(out['fsem'], y['fsem'], out, y, custom_loss(out, y)[0])
                pred[mode] = out['fsem']
            # agreement of the faults from both inputs with those from the image only
            scores['agreement'].add(pred['both'], (pred['image'] > 0.5).float())
            if i < opts.nplot:
                for mode in MODES:
                    write_array(get_numpy(pred[mode]), '%s/predict_%d.%s.fsem' % (opts.dir_output, i, mode))
                plot_sample(get_numpy(data['img']), get_numpy(data['meq']), get_numpy(label['fsem']), get_numpy(label['fsem_meq']),
                            {m: get_numpy(v) for m, v in pred.items()}, '%s/predict_%d.png' % (opts.dir_output, i))

        result = {m: s.result() for m, s in scores.items()}
        with open(opts.dir_output + '/metrics_validation.json', 'w') as f:
            json.dump(result, f, indent=1)
        with open(opts.dir_output + '/metrics_validation.txt', 'w') as f:
            f.write('%-20s %9s %9s %9s %11s %14s %9s\n' % ('mode', 'precision', 'recall', 'f1', 'dip_error', 'strike_error', 'loss'))
            for mode in MODES:
                s = result[mode]
                f.write('%-20s %9.4f %9.4f %9.4f %11.2f %14.2f %9.4f\n' % (TITLES[mode], s['precision'], s['recall'], s['f1'],
                                                                            s.get('dip_error', np.nan), s.get('strike_error', np.nan), s['loss']))
            f.write('F1 of the faults from both inputs against those from the image only: %.4f\n' % result['agreement']['f1'])
        plot_scores(result, opts.dir_output + '/metrics_validation.png')
        print(open(opts.dir_output + '/metrics_validation.txt').read())
        print(date_time(), ' >> Validation finished')

    if opts.model is not None:
        ## Inference

        net = load_model(opts.model)
        n1, n2, n3 = opts.n1, opts.n2, opts.n3
        shape = (1, 1, n1, n2, n3)
        assert opts.output is not None, 'inference needs --output'
        assert opts.input is not None or opts.input_meq is not None or opts.input_events is not None, \
            'inference needs an image (--input), a seismicity image (--input_meq) or events (--input_events)'

        # The volume stays in the CPU memory, and only the blocks go to the GPU
        img = read_array(opts.input, shape) if opts.input is not None else torch.zeros(shape)
        if opts.input_events is not None:
            ev = read_events(opts.input_events, 3)[0]
            meq = torch.from_numpy(render_seismicity_nd([ev[:, 0], ev[:, 1], ev[:, 2]], (n1, n2, n3)))[None, None]
        elif opts.input_meq is not None:
            meq = read_array(opts.input_meq, shape).clamp(0, 1)
        else:
            meq = torch.zeros(shape)
        img = normalize_image(img)

        for mode in (MODES if opts.mode == 'all' else [opts.mode]):
            x = {'img': torch.zeros_like(img) if mode == 'meq' else img, 'meq': torch.zeros_like(meq) if mode == 'image' else meq}
            out = predict(net, x, block=(opts.b1, opts.b2, opts.b3), overlap=(opts.p1, opts.p2, opts.p3))
            prefix = opts.output + ('.' + mode if opts.mode == 'all' else '')
            for k in ['fsem', 'fdip', 'fstrike']:
                write_array(get_numpy(out[k]), prefix + '.' + k)
            print(date_time(), ' >> %s: %s.fsem, .fdip, .fstrike' % (TITLES[mode], prefix))

        print(date_time(), ' >> Inference finished')

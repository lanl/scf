#
# (c) 2025. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by
# Triad National Security, LLC for the U.S. Department of Energy/National
# Nuclear Security Administration. All rights in the program are reserved
# by Triad National Security, LLC, and the U.S. Department of Energy/
# National Nuclear Security Administration.
# The Government is granted for itself and others acting on its behalf a nonexclusive,
# paid-up, irrevocable worldwide license in this material to reproduce, prepare,
# derivative works, distribute copies to the public, perform publicly
# and display publicly, and to permit others to do so.
#
# Author:
#   Kai Gao, kaigao@lanl.gov
#

import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import sys
import os
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import torchmetrics.functional as mf

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# sys.path.append(parent_dir)

from utility import *
from model2 import *

#==============================================================================
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--ntrain', type=int, default=1000, help='training size')
parser.add_argument('--nvalid', type=int, default=100, help='validation size')
parser.add_argument('--batch_train', type=int, default=1, help='training batch size')
parser.add_argument('--batch_valid', type=int, default=1, help='validation batch size')
parser.add_argument('--warmup', type=int, default=10, help='warmup epochs')
parser.add_argument('--epochs', type=int, default=100, help='max number of epochs')
parser.add_argument('--lr', type=float, default=0.5e-4, help='learning rate')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader')
parser.add_argument('--dir_output', type=str, default='./result', help='directory')
parser.add_argument('--dir_data_train', type=str, default='./dataset/data_train', help='directory')
parser.add_argument('--dir_target_train', type=str, default='./dataset/target_train', help='directory')
parser.add_argument('--dir_data_valid', type=str, default='./dataset/data_valid', help='directory')
parser.add_argument('--dir_target_valid', type=str, default='./dataset/target_valid', help='directory')
parser.add_argument('--resume', type=str, default=None, help='restart training from resume checkopoint')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus_per_node', type=int, default=4, help='number of gpus per node')
parser.add_argument('--seed', type=int, default=12345, help='random seed for initialization')
parser.add_argument('--check', type=str, default=None, help='name of the checkpoint')
parser.add_argument('--n1', '-n1', type=int, default=256, help='number of sampling points in x1')
parser.add_argument('--n2', '-n2', type=int, default=256, help='number of sampling points in x2')
parser.add_argument('--input', '-in', type=str, default=None, help='name of the input')
parser.add_argument('--model', '-model', type=str, default=None, help='name of the trained model')
parser.add_argument('--output', '-out', type=str, default=None, help='name of the output')
parser.add_argument('--finetune', type=str, default=None, help='name of the pre-trained model for fine tuning')
parser.add_argument('--b1', '-b1', type=int, default=None, help='inference phase block size in x1')
parser.add_argument('--b2', '-b2', type=int, default=None, help='inference phase block size in x2')
parser.add_argument('--p1', '-p1', type=int, default=32, help='inference phase block overlapping in x1')
parser.add_argument('--p2', '-p2', type=int, default=64, help='inference phase block overlapping in x2')
parser.add_argument('--pp', '-pp', type=str2bool, default='y', help='input PP image')
parser.add_argument('--ps', '-ps', type=str2bool, default='y', help='input PS image')
parser.add_argument('--sp', '-sp', type=str2bool, default='y', help='input SP image')
parser.add_argument('--ss', '-ss', type=str2bool, default='y', help='input SS image')
parser.add_argument('--net', '-net', type=str, default='scf', help='network name')
parser.add_argument('--precision', '-precision', type=str, default='32', 
                    help='precision of training; 16-mixed, bf16-mixed, or 32')
parser.add_argument('--plot', '-plot', type=str2bool, default='y', help='display validation results or not')
opts = parser.parse_args()

# Assert meaningful dimenisons
assert opts.n1 >= 1
assert opts.n2 >= 1

if opts.b1 is not None:
    assert opts.p1 < opts.b1
if opts.b2 is not None:
    assert opts.p2 < opts.b2

# Count number of input images
if opts.net == 'scf-elastic':
    nc = int(opts.pp) + int(opts.ps) + int(opts.sp) + int(opts.ss)

# Find device
if torch.cuda.is_available() and opts.gpus_per_node >= 1:
    device = torch.device('cuda')
    print(date_time(), ' >> Using GPU')
else:
    device = torch.device('cpu')
    print(date_time(), ' >> Using CPU')

# Set precision
torch.set_float32_matmul_precision('high')


#==============================================================================
class BasicDataset(Dataset):

    def __init__(self, dir_data, dir_target, data_ids, dim=(opts.n1, opts.n2)):
        self.dir_data = dir_data
        self.dir_target = dir_target
        self.ids = data_ids
        self.n1 = dim[0]
        self.n2 = dim[1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):

        idx = str(self.ids[i])

        # Read data
        data = {}

        if opts.net == 'mtl':
            data['img'] = read_array(self.dir_data + '/' + idx + '_img.bin', (1, self.n1, self.n2))
            data['meq'] = torch.empty(0)

        if opts.net == 'mtl-seismicity':
            img = read_array(self.dir_data + '/' + idx + '_img.bin', (1, self.n1, self.n2))
            meq = read_array(self.dir_data + '/' + idx + '_meq.bin', (1, self.n1, self.n2))
            data['img'] = torch.cat((img, meq), dim=0)
            data['meq'] = torch.empty(0)

        if opts.net == 'scf':
            data['img'] = read_array(self.dir_data + '/' + idx + '_img.bin', (1, self.n1, self.n2))
            data['meq'] = read_array(self.dir_data + '/' + idx + '_meq.bin', (1, self.n1, self.n2))

        if opts.net == 'scf-elastic':
            if opts.pp:
                pp = read_array(self.dir_data + '/' + idx + '_img_pp.bin', (1, self.n1, self.n2))
            else:
                pp = None
            if opts.ps:
                ps = read_array(self.dir_data + '/' + idx + '_img_ps.bin', (1, self.n1, self.n2))
            else:
                ps = None
            if opts.sp:
                sp = read_array(self.dir_data + '/' + idx + '_img_sp.bin', (1, self.n1, self.n2))
            else:
                sp = None
            if opts.ss:
                ss = read_array(self.dir_data + '/' + idx + '_img_ss.bin', (1, self.n1, self.n2))
            else:
                ss = None
            data['img'] = torch.cat([t for t in [pp, ps, sp, ss] if t is not None], dim=0)
            data['meq'] = read_array(self.dir_data + '/' + idx + '_meq.bin', (1, self.n1, self.n2))

        # Read label
        target = {}
        target['fsem'] = read_array(self.dir_target + '/' + idx + '_fsem.bin', (1, self.n1, self.n2))
        target['fdip'] = read_array(self.dir_target + '/' + idx + '_fdip.bin', (1, self.n1, self.n2))

        return data, target


def custom_loss(y_pred, y_true):

    # fault probability
    mp = y_pred['fsem']
    mt = y_true['fsem']
    loss_fsem = 1.0 - (2.0 * torch.sum(mp * mt) + 1.0) / (torch.sum(mp + mt) + 1.0)

    # fault dip
    dp = y_pred['fdip']
    dt = y_true['fdip']
    loss_fdip = F.l1_loss(dp, dt) * 10

    loss = loss_fsem + loss_fdip

    return loss, loss_fsem, loss_fdip


def custom_accuracy(y_pred, y_true):

    # accuracy
    accuracy = mf.classification.binary_accuracy(y_pred, y_true)

    # ssim
    s = ssim(y_pred, y_true, data_range=1.0)

    # precision and recall
    precision = mf.classification.binary_precision(y_pred, y_true)
    recall = mf.classification.binary_recall(y_pred, y_true)

    return accuracy, precision, recall, s


#==============================================================================
class mtlnet(pl.LightningModule):

    def __init__(self, lr: float = 1.0e-4, warmup_epochs: int=10, max_epochs: int=100):

        super(mtlnet, self).__init__()
        
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.lr = lr
        self.l1 = 32
        self.l2 = 64
        self.l3 = 128

        if opts.net == 'mtl' or opts.net == 'mtl-seismicity':

            if opts.net == 'mtl':
                self.in_channels = 1
            else:
                self.in_channels = 2

            self.encoder1 = resu1(self.in_channels, self.l1)
            self.encoder2 = resu2(self.l1, self.l2)
            self.encoder3 = resu3(self.l2, self.l3)
            self.decoder = mtl_decoder_nofusion(self.l1, self.l2, self.l3, out_ch=self.l1, last_kernel_size=3)

        if opts.net == 'scf' or opts.net == 'scf-elastic':

            if opts.net == 'scf':
                self.in_channels = 1
            else:
                self.in_channels = nc

            self.encoder1_img = resu1(self.in_channels, self.l1)
            self.encoder2_img = resu2(self.l1, self.l2)
            self.encoder1_meq = resu1(1, self.l1)
            self.encoder2_meq = resu2(self.l1, self.l2)

            self.fusion3 = SpatialChannelAttention((self.l2, self.l2), 2*self.l2, channel_ratio=8)
            self.encoder3 = resu3(2 * self.l2, self.l3)

            self.decoder = mtl_decoder_fusion(self.l1, self.l2, self.l3, out_ch=self.l1, last_kernel_size=3)

        self.subdecoder_fault_semantic = mtl_subdecoder(in_ch=self.l1,
                                                        out_ch=1,
                                                        bn=None,
                                                        mid_activation='relu',
                                                        activation='sigmoid')
        self.subdecoder_fault_dip = mtl_subdecoder(in_ch=self.l1,
                                                   out_ch=1,
                                                   bn='instance',
                                                   mid_activation='relu',
                                                   activation='sigmoid')

    def forward(self, x):

        if opts.net == 'mtl' or opts.net == 'mtl-seismicity':

            # Migration image encoder
            out_encoder1 = self.encoder1(x['img'])
            out_encoder2 = self.encoder2(maxpool(out_encoder1, 2))
            out_encoder3 = self.encoder3(maxpool(out_encoder2, 2))

            # Decoders
            out_fault = self.decoder(x['img'], out_encoder1, out_encoder2, out_encoder3)

        if opts.net == 'scf' or opts.net == 'scf-elastic':

            # Migration image encoder
            out_encoder1_img = self.encoder1_img(x['img'])
            out_encoder2_img = self.encoder2_img(maxpool(out_encoder1_img, 2))

            # Seismicity image encoder
            out_encoder1_meq = self.encoder1_meq(x['meq'])
            out_encoder2_meq = self.encoder2_meq(maxpool(out_encoder1_meq, 2))

            # Bottleneck encoder
            f3 = self.fusion3(maxpool(out_encoder2_img, 2), maxpool(out_encoder2_meq, 2))
            out_encoder3 = self.encoder3(f3)

            # Decoders
            out_fault = self.decoder(x['img'], out_encoder1_img, out_encoder2_img, out_encoder1_meq, out_encoder2_meq, out_encoder3)

        # Subdecoders
        out_fault_semantic = self.subdecoder_fault_semantic(out_fault)
        out_fault_dip = self.subdecoder_fault_dip(out_fault) * out_fault_semantic
        out = {}
        out['fsem'] = out_fault_semantic
        out['fdip'] = out_fault_dip

        return out

    def training_step(self, batch):

        x, y_true = batch
        y_pred = self.forward(x)

        loss, loss_fault_semantic, loss_fault_dip = custom_loss(y_pred, y_true)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_fsem", loss_fault_semantic, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss_fdip", loss_fault_dip, on_step=False, on_epoch=True, prog_bar=False)

        accuracy, precision, recall, s = custom_accuracy(y_pred['fsem'], y_true['fsem'])
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_precision", precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_recall", recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_ssim", s, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch):

        x, y_true = batch
        y_pred = self.forward(x)

        loss, loss_fault_semantic, loss_fault_dip = custom_loss(y_pred, y_true)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_loss_fsem", loss_fault_semantic, on_step=False, on_epoch=True, prog_bar=False)
        self.log("valid_loss_fdip", loss_fault_dip, on_step=False, on_epoch=True, prog_bar=False)

        accuracy, precision, recall, s = custom_accuracy(y_pred['fsem'], y_true['fsem'])
        self.log("valid_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("valid_precision", precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("valid_recall", recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log("valid_ssim", s, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=1e-4)

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return float(epoch + 1) / float(self.warmup_epochs)
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }



#==============================================================================
if __name__ == '__main__':

    if opts.input is None and opts.check is None:
        ## Training phase

        logger = TensorBoardLogger(opts.dir_output)
        checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                              dirpath=opts.dir_output,
                                              filename='{epoch:d}',
                                              mode='min',
                                              save_top_k=opts.epochs,
                                              save_last=True)

        lr_monitor = LearningRateMonitor(logging_interval='step')

        params = {
            'max_epochs': opts.epochs,
            'default_root_dir': opts.dir_output,
            'logger': logger,
            'callbacks': [checkpoint_callback, lr_monitor],
            'precision': opts.precision
        }

        if torch.cuda.is_available() and opts.gpus_per_node >= 1:
            params['devices'] = opts.gpus_per_node
            params['num_nodes'] = opts.nodes
            params['accelerator'] = 'gpu'
            params['strategy'] = 'ddp'

        trainer = pl.Trainer(**params)

        t = BasicDataset(opts.dir_data_train,
                         opts.dir_target_train,
                         data_ids=np.arange(0, opts.ntrain),
                         dim=(opts.n1, opts.n2))
        train_loader = DataLoader(t, batch_size=opts.batch_train, num_workers=opts.threads, shuffle=True)

        v = BasicDataset(opts.dir_data_valid,
                         opts.dir_target_valid,
                         data_ids=np.arange(0, opts.nvalid),
                         dim=(opts.n1, opts.n2))
        valid_loader = DataLoader(v, batch_size=opts.batch_valid, num_workers=opts.threads)

        set_random_seed(opts.seed)
        params = {'lr': opts.lr, 
                  'warmup_epochs': opts.warmup,
                  'max_epochs': opts.epochs}
        net = mtlnet(**params)
        
        # If fine tuning, then read the pre-trained model
        if opts.finetune is not None:
            net.load_state_dict(torch.load(opts.finetune, map_location=device)['state_dict'])
            net.to(device)

        # If resuming from a breakpoint, read the breakpoint model
        if opts.resume:
            trainer.fit(net, train_loader, valid_loader, ckpt_path=opts.resume)
        else:
            trainer.fit(net, train_loader, valid_loader)

        print(date_time(), ' >> Training finished')

    if opts.input is None and opts.check is not None:
        ## Validation phase
        
        set_font()

        v = BasicDataset(opts.dir_data_valid,
                         opts.dir_target_valid,
                         data_ids=np.arange(0, opts.nvalid),
                         dim=(opts.n1, opts.n2))
        valid_loader = DataLoader(v, batch_size=opts.batch_valid, num_workers=opts.threads)

        net = mtlnet()
        net.load_state_dict(torch.load(opts.check, map_location=device)['state_dict'])
        net.to(device)
        
        # Plot the validation results
        l = 1
        loss_average = 0
        loss_fault_semantic_average = 0
        loss_fault_dip_average = 0
        accuracy_average = 0
        precision_average = 0
        recall_average = 0
        ssim_average = 0
        with tqdm(total=len(v), desc='', unit='image') as pbar:

            for (input, target) in valid_loader:
                
                target['fsem'] = target['fsem'].to(device)
                target['fdip'] = target['fdip'].to(device)

                with torch.no_grad():
                    input['img'] = input['img'].to(device)
                    input['meq'] = input['meq'].to(device)
                    predict = net(input)

                # Metrics     
                d1, d2, d3 = custom_loss(predict, target)
                a1, a2, a3, a4 = custom_accuracy(predict['fsem'], target['fsem'])
                loss_average = loss_average + d1
                loss_fault_semantic_average = loss_fault_semantic_average + d2
                loss_fault_dip_average = loss_fault_dip_average + d3
                accuracy_average = accuracy_average + a1
                precision_average = precision_average + a2
                recall_average = recall_average + a3
                ssim_average = ssim_average + a4
                    
                # For each batch
                bs = opts.batch_valid
                for i in range(0, bs):
                    
                    # Save prediction
                    ir = (l - 1) * opts.batch_valid + i
                    fsem = get_numpy(predict['fsem'][i])
                    fdip = get_numpy(predict['fdip'][i])
                    write_array(fsem, opts.dir_output + '/predict_' + str(ir) + '_fsem.bin')
                    write_array(fdip, opts.dir_output + '/predict_' + str(ir) + '_fdip.bin')
                    
                    if opts.plot:
                    
                        # Plot
                        if opts.net == 'mtl':
                            im = get_numpy(input['img'][i])
                            mq = np.zeros_like(im)
                            
                        if opts.net == 'mtl-seismicity':
                            im = get_numpy(input['img'][i][0])
                            mq = get_numpy(input['img'][i][1])
                            
                        if opts.net == 'scf':
                            im = get_numpy(input['img'][i])
                            mq = get_numpy(input['meq'][i])
                            
                        if opts.net == 'scf-elastic':
                            im = get_numpy(input['img'][i][0])
                            # im = np.transpose(im, (1, 0, 2)).reshape(opts.n1, opts.n2 * nc)
                            mq = get_numpy(input['meq'][i])
    
                        # Plot input and output
                        fig, ax = plt.subplots(2, 3, figsize=(9.5, 6))
    
                        plot_img = ax[0][0].imshow(im, cmap='gray', vmin=-3, vmax=3)
                        ax[0][0].set_title('Input - Image')
                        if opts.net != 'mtl':
                            plot_meq = ax[1][0].imshow(mq, cmap='jet', vmin=0, vmax=1)
                            ax[1][0].set_title('Input - Seismicity')
                        else:
                            ax[1][0].axis('off')
                        
                        fsem = get_numpy(target['fsem'][i])
                        fdip = get_numpy(target['fdip'][i])
                        plot_fsem = ax[0][1].imshow(fsem, cmap='viridis', vmin=0, vmax=1)
                        ax[0][1].set_title('Probability - GT')
                        plot_fdip = ax[0][2].imshow(fdip, cmap='jet', vmin=0, vmax=1)
                        ax[0][2].set_title('Dip - GT')
                        
                        fsem = get_numpy(predict['fsem'][i])
                        fdip = get_numpy(predict['fdip'][i])
                        ax[1][1].imshow(fsem, cmap='viridis', vmin=0, vmax=1)
                        ax[1][1].set_title('Probability - Prediction')
                        ax[1][2].imshow(fdip, cmap='jet', vmin=0, vmax=1)
                        ax[1][2].set_title('Dip - Prediction')
                        
                        for ii in range(2):
                            for jj in range(3):
                        
                                # Move x-axis labels to top
                                ax[ii][jj].xaxis.tick_top()
                                ax[ii][jj].xaxis.set_label_position('top')
                            
                                # Set major and minor ticks
                                ax[ii][jj].xaxis.set_major_locator(MultipleLocator(100))
                                ax[ii][jj].xaxis.set_minor_locator(MultipleLocator(10))
                                ax[ii][jj].yaxis.set_major_locator(MultipleLocator(100))
                                ax[ii][jj].yaxis.set_minor_locator(MultipleLocator(10))
    
                        # Add 3 vertical colorbars evenly spaced vertically on the right
                        if opts.net != 'mtl':
                            
                            # Adjust left, bottom, width, height of each
                            cb_width = 0.015
                            cb_left = 0.92
                            cb_height = 0.19
                            spacing = 0.025
                            
                            bottoms = [0.725, 0.5, 0.275, 0.05]
                            
                            cb_axes = [fig.add_axes([cb_left, b, cb_width, cb_height]) for b in bottoms]
                            
                            # Add colorbars with custom titles
                            cb1 = fig.colorbar(plot_img, cax=cb_axes[0])
                            cb2 = fig.colorbar(plot_meq, cax=cb_axes[1])
                            cb3 = fig.colorbar(plot_fsem, cax=cb_axes[2])
                            cb4 = fig.colorbar(plot_fdip, cax=cb_axes[3])
                            
                            cb1.set_label("Image Amplitude", rotation=90, labelpad=10)
                            cb2.set_label("Seismicity", rotation=90, labelpad=10)
                            cb3.set_label("Probability", rotation=90, labelpad=10)
                            cb4.set_label("Dip", rotation=90, labelpad=10)
                            
                            cb1.locator = MultipleLocator(1)
                            cb1.update_ticks()
                            cb2.locator = MultipleLocator(0.5)
                            cb2.update_ticks()
                            cb3.locator = MultipleLocator(0.5)
                            cb3.update_ticks()
                            cb4.locator = MultipleLocator(0.5)
                            cb4.update_ticks()
                            
                            for label in cb1.ax.get_yticklabels():
                                label.set_rotation(90)
                            for label in cb2.ax.get_yticklabels():
                                label.set_rotation(90)
                            for label in cb3.ax.get_yticklabels():
                                label.set_rotation(90)
                            for label in cb4.ax.get_yticklabels():
                                label.set_rotation(90)
                            
                        else:
                            
                            # Adjust left, bottom, width, height of each
                            cb_width = 0.015
                            cb_left = 0.92
                            cb_height = 0.25  # ~1/3 of the figure
                            spacing = 0.05    # vertical spacing between them
                            
                            bottoms = [0.65, 0.35, 0.05]  # Positions for 3 colorbars
                            
                            cb_axes = [fig.add_axes([cb_left, b, cb_width, cb_height]) for b in bottoms]
                            
                            # Add colorbars with custom titles
                            cb1 = fig.colorbar(plot_img, cax=cb_axes[0])
                            cb2 = fig.colorbar(plot_fsem, cax=cb_axes[1])
                            cb3 = fig.colorbar(plot_fdip, cax=cb_axes[2])
                            
                            cb1.set_label("Image Amplitude", rotation=90, labelpad=10)
                            cb2.set_label("Probability", rotation=90, labelpad=10)
                            cb3.set_label("Dip", rotation=90, labelpad=10)
                            
                            cb1.locator = MultipleLocator(1)
                            cb1.update_ticks()
                            cb2.locator = MultipleLocator(0.5)
                            cb2.update_ticks()
                            cb3.locator = MultipleLocator(0.5)
                            cb3.update_ticks()
                            
                            for label in cb1.ax.get_yticklabels():
                                label.set_rotation(90)
                            for label in cb2.ax.get_yticklabels():
                                label.set_rotation(90)
                            for label in cb3.ax.get_yticklabels():
                                label.set_rotation(90)
                                        
                        plt.tight_layout(rect=[0, 0, 0.9, 1])

                        plt.savefig(opts.dir_output + '/predict_' + str(ir) + '.pdf', dpi=300, bbox_inches='tight', pad_inches=1.0/72.0)
                        # plt.show()
                        plt.close()

                pbar.update(bs)
                l = l + 1

        # Output validation set metrics
        l = l - 1
        loss_average = loss_average / l
        loss_fault_semantic_average = loss_fault_semantic_average / l
        loss_fault_dip_average = loss_fault_dip_average / l
        accuracy_average = accuracy_average / l
        precision_average = precision_average / l
        recall_average = recall_average / l
        ssim_average = ssim_average / l

        with open(opts.dir_output + '/metrics_validation.txt', 'w') as f:
            f.write('loss' + '\t' + str(get_numpy(loss_average)) + '\n')
            f.write('loss_fault_semantic' + '\t' + str(get_numpy(loss_fault_semantic_average)) + '\n')
            f.write('loss_fault_dip' + '\t' + str(get_numpy(loss_fault_dip_average)) + '\n')
            f.write('accuracy' + '\t' + str(get_numpy(accuracy_average)) + '\n')
            f.write('precision' + '\t' + str(get_numpy(precision_average)) + '\n')
            f.write('recall' + '\t' + str(get_numpy(recall_average)) + '\n')
            f.write('ssim' + '\t' + str(get_numpy(ssim_average)) + '\n')
            f.close()

        print(date_time(), ' >> Validation finished')

    if opts.input is not None:
        ## Inference phase

        # Read image
        n1 = opts.n1
        n2 = opts.n2

        # # Load trained model
        net = mtlnet()
        net.load_state_dict(torch.load(opts.model, map_location=device)['state_dict'])
        net.to(device)

        print(date_time(), ' >> Pretrained model loaded')

        b1 = opts.b1
        b2 = opts.b2
    
        if b1 is None and b2 is None:
            # Infer using the entire seismic image and source image
            
            data = {}
            
            if opts.net == 'mtl':
                data['img'] = read_array(opts.input + '.img', (1, 1, n1, n2))
                data['meq'] = torch.empty(0)
            
            if opts.net == 'mtl-seismicity':
                x1 = read_array(opts.input + '.img', (1, 1, n1, n2))
                x2 = read_array(opts.input + '.meq', (1, 1, n1, n2))
                data['img'] = torch.cat((x1, x2), dim=0)
                data['meq'] = torch.empty(0)
            
            if opts.net == 'scf':
                data['img'] = read_array(opts.input + '.img', (1, 1, n1, n2))
                data['meq'] = read_array(opts.input + '.meq', (1, 1, n1, n2))
            
            if opts.net == 'scf-elastic':
                if opts.pp:
                    pp = read_array(opts.input + '.img.pp', (1, 1, n1, n2))
                if opts.ps:
                    ps = read_array(opts.input + '.img.ps', (1, 1, n1, n2))
                if opts.sp:
                    sp = read_array(opts.input + '.img.sp', (1, 1, n1, n2))
                if opts.ss:
                    ss = read_array(opts.input + '.img.ss', (1, 1, n1, n2))
                data['img'] = torch.cat([t for t in [pp, ps, sp, ss] if t is not None], dim=1)
                data['meq'] = read_array(opts.input + '.meq', (1, 1, n1, n2))
                
            data['img'] = data['img'].to(device)
            data['meq'] = data['meq'].to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    predict = net(data)
    
        else:
            # For large models, infer block by block
                
            img = None
            meq = None
            pp = None
            ps = None
            sp = None
            ss = None
                        
            if opts.net == 'mtl':
                img = read_array(opts.input + '.img', (1, 1, n1, n2))

            if opts.net == 'mtl-seismicity':
                img = read_array(opts.input + '.img', (1, 1, n1, n2))
                meq = read_array(opts.input + '.meq', (1, 1, n1, n2))
            
            if opts.net == 'scf':
                img = read_array(opts.input + '.img', (1, 1, n1, n2))
                meq = read_array(opts.input + '.meq', (1, 1, n1, n2))

            if opts.net == 'scf-elastic':
                if opts.pp:
                    pp = read_array(opts.input + '.img.pp', (1, 1, n1, n2))
                if opts.ps:
                    ps = read_array(opts.input + '.img.ps', (1, 1, n1, n2))
                if opts.sp:
                    sp = read_array(opts.input + '.img.sp', (1, 1, n1, n2))
                if opts.ss:
                    ss = read_array(opts.input + '.img.ss', (1, 1, n1, n2))
                meq = read_array(opts.input + '.meq', (1, 1, n1, n2))
                
            for i in [img, meq, pp, ps, sp, ss]:
                if i is not None:
                    i = F.pad(i, [0, n2 - opts.n2, 0, n1 - opts.n1], mode='reflect')

            predict = {}
            predict['fsem'] = torch.zeros((1, 1, n1, n2))
            predict['fdip'] = torch.zeros((1, 1, n1, n2))
            predict['fstrike'] = torch.zeros((1, 1, n1, n2))
            
            if b1 is None:
                b1 = n1
            if b2 is None:
                b2 = n2
    
            p1 = opts.p1
            p2 = opts.p2
            
            nb1, s1, e1 = rollover_block(n1, b1, p1, allow_longer=True)
            nb2, s2, e2 = rollover_block(n2, b2, p2, allow_longer=True)
            
            n1 = e1[-1] + 1
            n2 = e2[-1] + 1
            if b1 is None:
                b1 = n1
            if b2 is None:
                b2 = n2
            
            for i in range(nb1):
                for j in range(nb2):
                        
                    # Note Python's a:b is actually a..b - 1 in other languages
                    i1 = s1[i]
                    h1 = e1[i] + 1
                    i2 = s2[j]
                    h2 = e2[j] + 1
                    
                    print(date_time(), ' >> Block index = ', 
                          str(i + 1) + '/' + str(nb1) + ', ', 
                          str(j + 1) + '/' + str(nb2))
                    print(date_time(), ' >> Block range = ', 
                          str(i1 + 1) + '-' + str(h1 - 1 + 1) + ', ', 
                          str(i2 + 1) + '-' + str(h2 - 1 + 1))
                    
                    # Get data
                    data = {}

                    if opts.net == 'mtl':
                        data['img'] = img[:, :, i1:h1, i2:h2]
                        data['meq'] = torch.empty(0)
                    
                    if opts.net == 'mtl-seismicity':
                        x1 = img[:, :, i1:h1, i2:h2]
                        x2 = meq[:, :, i1:h1, i2:h2]
                        data['img'] = torch.cat((x1, x2), dim=0)
                        data['meq'] = torch.empty(0)
                    
                    if opts.net == 'scf':
                        data['img'] = img[:, :, i1:h1, i2:h2]
                        data['meq'] = meq[:, :, i1:h1, i2:h2]
                    
                    if opts.net == 'scf-elastic':
                        if opts.pp:
                            pp = pp[:, :, i1:h1, i2:h2]
                        if opts.ps:
                            ps = ps[:, :, i1:h1, i2:h2]
                        if opts.sp:
                            sp = sp[:, :, i1:h1, i2:h2]
                        if opts.ss:
                            ss = ss[:, :, i1:h1, i2:h2]
                        data['img'] = torch.cat([t for t in [pp, ps, sp, ss] if t is not None], dim=1)
                        data['meq'] = meq[:, :, i1:h1, i2:h2]
                        
                    data['img'] = data['img'].to(device)
                    data['meq'] = data['meq'].to(device)

                    # Inference for this block
                    with torch.no_grad():
                        with torch.autocast(device_type=device.type, dtype=torch.float16):
                            predict_block = net(data)

                    # Merge to whole
                    predict['fsem'] = merge_block_2d(predict['fsem'], 
                                                     taper_2d_bc(predict_block['fsem'].to('cpu'), [1, 1, 1, 1], method='zero'), 
                                                     range=(i1, h1, i2, h2), 
                                                     mode='signed_max')
                    predict['fdip'] = merge_block_2d(predict['fdip'], 
                                                     taper_2d_bc(predict_block['fdip'].to('cpu'), [1, 1, 1, 1], method='zero'), 
                                                     range=(i1, h1, i2, h2), 
                                                     mode='signed_max')

        ## Output inference results
        write_array(get_numpy(predict['fsem']), opts.output + '.fsem')
        write_array(get_numpy(predict['fdip']), opts.output + '.fdip')

        print(date_time(), ' >> Inference finished')
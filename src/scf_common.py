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
# Shared building blocks for training, validating and applying SCF-Net in 2D and 3D: amplitude scale, input
# modes, random training crops, losses and metrics, and block-wise inference. All tensors are (N, C, n1, n2)
# in 2D or (N, C, n1, n2, n3) in 3D, with n1 the depth axis.
#

import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

# Total pooling factor of SCF-Net: an output pixel does not change when the input is shifted by a multiple of
# this factor, so training crops and inference blocks start at multiples of it
POOL = 8


# ==============================================================================
# Amplitude scale
# ==============================================================================
def estimate_scale(x, eps=1.0e-12):
    """
    Root-mean-square amplitude (N, 1, 1, ...) of the nonzero samples of every sample of x (N, C, ...), so that dead
    traces and muted parts do not lower the scale.
    """
    n = x.shape[0]
    flat = x.reshape(n, -1)
    nz = (flat != 0).sum(dim=1).clamp_min(1)
    s = (flat.pow(2).sum(dim=1) / nz).sqrt().clamp_min(eps)
    return s.view(n, *([1] * (x.dim() - 1)))


def gaussian_smooth(x, sigma):
    """Separable Gaussian smoothing of x (N, C, n1, n2) or (N, C, n1, n2, n3) with replicate padding."""
    r = int(math.ceil(3.0 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma)**2)
    k = k / k.sum()
    c = x.shape[1]
    nd = x.dim() - 2
    conv = F.conv2d if nd == 2 else F.conv3d
    for ax in range(nd):
        shape = [1] * nd
        shape[ax] = -1
        pad = [0] * (2 * nd)
        pad[2 * (nd - 1 - ax)] = r
        pad[2 * (nd - 1 - ax) + 1] = r
        x = conv(F.pad(x, pad, mode='replicate'), k.view(1, 1, *shape).repeat(c, *([1] * (nd + 1))), groups=c)
    return x


def agc(x, sigma=32.0, floor=0.1):
    """
    Automatic gain control of an image x (1, 1, n1, n2[, n3]): the image of unit RMS divided by its local RMS, from a
    Gaussian window of standard deviation sigma, with a floor of `floor` times the global RMS so that quiet parts
    are not amplified to the level of the reflections. Apply it to the whole image before block-wise inference.
    """
    x = x / estimate_scale(x)
    local = gaussian_smooth(x.pow(2), sigma).sqrt()
    return x / torch.sqrt(local**2 + floor**2)


# ==============================================================================
# Input modes
# ==============================================================================
def apply_mode(data, target, mode):
    """
    Inputs and labels of a sample for an input mode:
    both:  image and seismicity; the label is all faults
    image: the seismicity is set to zero; the label is all faults
    meq:   the image is set to zero; the label is the part of the faults that the seismicity indicates
           ('fsem_meq'), since faults without seismicity cannot be found from it
    """
    data = dict(data)
    target = dict(target)
    if mode == 'image':
        data['meq'] = torch.zeros_like(data['meq'])
    elif mode == 'meq':
        data['img'] = torch.zeros_like(data['img'])
        target['fsem'] = target['fsem_meq']
    elif mode != 'both':
        raise ValueError('unknown mode: ' + str(mode))
    return data, target


def random_mode(p_image, p_meq):
    """Draw an input mode: 'image' with probability p_image, 'meq' with p_meq, otherwise 'both'."""
    r = random.random()
    if r < p_image:
        return 'image'
    if r < p_image + p_meq:
        return 'meq'
    return 'both'


# ==============================================================================
# Random training crops
# ==============================================================================
class CropCollate:
    """
    Collate function of a training DataLoader (a class, so that DataLoader workers can be started by fork,
    forkserver or spawn).

    Every batch is cropped to one random size that is a multiple of `multiple`, at least min_size and at most
    max_size (0: the sample size) along each axis (min_size 0: the axis is not cropped), and every sample at its
    own random position, so the network does not learn the position of a pixel from its distance to the image
    boundary. The image of every sample is multiplied by a random gain, log-uniform in `gain`, and its polarity is
    reversed with probability `flip`, so the network tolerates an amplitude scale that differs from that of the
    training images. The seismicity location image of half of the samples is blurred by a Gaussian of a standard
    deviation drawn from `meq_blur`, with the peak of an isolated event restored to one, so the network tolerates
    seismicity images rendered with wider Gaussians. Every sample then gets a random input mode (apply_mode,
    random_mode). Items are (data, target) dictionaries of tensors (C, n1, n2[, n3]).
    """

    def __init__(self, min_size, max_size=None, p_image=0.0, p_meq=0.0, gain=(1.0, 1.0), flip=0.0, meq_blur=(0.0, 0.0),
                 multiple=POOL):
        self.min_size = tuple(min_size)
        self.max_size = None if max_size is None else tuple(max_size)
        self.p_image = p_image
        self.p_meq = p_meq
        self.gain = tuple(gain)
        self.flip = flip
        self.meq_blur = tuple(meq_blur)
        self.multiple = multiple

    def __call__(self, batch):
        m8 = self.multiple
        shape = tuple(batch[0][0]['img'].shape[1:])
        size = []
        for ax, (m, n) in enumerate(zip(self.min_size, shape)):
            hi = n if self.max_size is None or self.max_size[ax] <= 0 else min(n, self.max_size[ax])
            if m <= 0 or m >= hi:
                size.append(hi - hi % m8)
            else:
                size.append(m8 * random.randint(min(int(math.ceil(m / m8)), hi // m8), hi // m8))
        out = []
        for data, target in batch:
            start = [random.randint(0, n - s) for n, s in zip(shape, size)]
            sl = (slice(None),) + tuple(slice(a, a + s) for a, s in zip(start, size))
            d = {k: v[sl] for k, v in data.items()}
            t = {k: v[sl] for k, v in target.items()}
            g = math.exp(random.uniform(math.log(self.gain[0]), math.log(self.gain[1])))
            d['img'] = d['img'] * (-g if random.random() < self.flip else g)
            if self.meq_blur[1] > 0 and random.random() < 0.5:
                sb = random.uniform(self.meq_blur[0], self.meq_blur[1])
                if sb > 0.1:
                    # an isolated event exp(-0.3 r^2) has the variance s0 along every axis; the blur adds sb^2 and
                    # lowers its peak by (s0 / (s0 + sb^2))^(nd / 2) in nd dimensions
                    s0 = 1.0 / (2.0 * 0.3)
                    nd = d['meq'].dim() - 1
                    d['meq'] = (gaussian_smooth(d['meq'].unsqueeze(0), sb)[0] * ((s0 + sb**2) / s0)**(0.5 * nd)).clamp(0, 1)
            d, t = apply_mode(d, t, random_mode(self.p_image, self.p_meq))
            out.append((d, t))
        return default_collate(out)


def random_subcrop(shape, frac=(0.4, 0.75), min_size=64, multiple=POOL):
    """
    Slices of a random sub-crop of a crop of spatial shape `shape`, with a size of frac times the crop and a start
    on the pooling grid of the crop, so that the network computes the same features for the sub-crop and the
    crop away from the sub-crop edges.
    """
    sl = [slice(None), slice(None)]
    for n in shape:
        lo = max(min_size, int(math.ceil(frac[0] * n / multiple)) * multiple)
        hi = max(lo, int(frac[1] * n / multiple) * multiple)
        s = min(n, multiple * random.randint(lo // multiple, hi // multiple))
        a = multiple * random.randint(0, (n - s) // multiple)
        sl.append(slice(a, a + s))
    return tuple(sl)


# ==============================================================================
# Losses and metrics
# ==============================================================================
def dice_loss(p, t, eps=1.0):
    """Dice loss of a fault probability over a whole batch."""
    return 1.0 - (2.0 * torch.sum(p * t) + eps) / (torch.sum(p) + torch.sum(t) + eps)


def bce_loss(p, t, eps=1.0e-6):
    """Binary cross entropy of a probability, in float32."""
    p = p.float().clamp(eps, 1.0 - eps)
    t = t.float()
    return -torch.mean(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))


def kl_loss(p, q, eps=1.0e-6):
    """
    Mean binary Kullback-Leibler divergence of the probability p from the target probability q, in float32: zero if
    and only if p = q, with a gradient (with respect to the logit of p) of p - q that does not vanish where p
    saturates.
    """
    p = p.float().clamp(eps, 1.0 - eps)
    q = q.float().clamp(eps, 1.0 - eps)
    return torch.mean(q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p)))


def masked_l1(p, t, mask, eps=1.0):
    """Mean absolute difference of p and t over the pixels where mask = 1."""
    return torch.sum(torch.abs(p - t) * mask) / (torch.sum(mask) + eps)


def dilate(x, tol):
    """Binary dilation of x (N, 1, n1, n2[, n3]) with a square (cube) of 2 tol + 1 samples."""
    if tol <= 0:
        return x
    if x.dim() == 4:
        return F.max_pool2d(x, 2 * tol + 1, 1, tol)
    return F.max_pool3d(x, 2 * tol + 1, 1, tol)


def fault_scores(p, t, tol=2, threshold=0.5, eps=1.0e-6):
    """
    Precision, recall and F1 of the faults p > threshold against the labels t, where a predicted fault pixel within
    tol pixels of a true fault is correct, and a true fault pixel within tol pixels of a predicted fault is found.
    Computed over the whole batch; without predicted and true faults, all three are one.
    """
    pb = (p > threshold).float()
    tb = (t > 0.5).float()
    precision = (torch.sum(pb * dilate(tb, tol)) + eps) / (torch.sum(pb) + eps)
    recall = (torch.sum(tb * dilate(pb, tol)) + eps) / (torch.sum(tb) + eps)
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


# ==============================================================================
# Block-wise inference
# ==============================================================================
def block_starts(n, b, o, multiple=POOL):
    """
    Block size, start indices and padded size along one axis of n samples.

    b: block size (0 or >= n: the whole axis); o: overlap. The axis is padded at its end to a multiple of
    `multiple` (at most multiple - 1 samples), as for the whole image. The block size and the starts are multiples
    of `multiple`, so every block starts on the pooling grid of the whole image; the step between blocks is at most
    b - o, and the last block ends at the end of the padded axis.
    """
    total = int(math.ceil(n / multiple)) * multiple
    b = int(math.ceil(b / multiple)) * multiple
    if b <= 0 or b >= total:
        return total, [0], total
    step = max(multiple, ((b - o) // multiple) * multiple)
    starts = list(range(0, total - b, step)) + [total - b]
    return b, starts, total


def taper_weight(size, overlap, device):
    """Separable raised-cosine weight that ramps up over `overlap` samples at both ends of each axis."""
    w = None
    for ax, (n, o) in enumerate(zip(size, overlap)):
        r = torch.ones(n, device=device)
        o = min(o, n // 2)
        if o > 0:
            ramp = 0.5 - 0.5 * torch.cos(math.pi * (torch.arange(o, device=device) + 0.5) / o)
            r[:o] = ramp
            r[n - o:] = torch.flip(ramp, [0])
        shape = [1] * len(size)
        shape[ax] = n
        r = r.view(shape)
        w = r if w is None else w * r
    return w.view(1, 1, *size)


def pad_end(x, total):
    """Pad x (1, C, n1, n2[, n3]) at the end of each spatial axis to the sizes `total` (reflect if possible)."""
    pad = []
    for ax in reversed(range(len(total))):
        pad += [0, total[ax] - x.shape[2 + ax]]
    if all(p == 0 for p in pad):
        return x
    if all(pad[2 * i + 1] < x.shape[2 + len(total) - 1 - i] for i in range(len(total))):
        return F.pad(x, pad, mode='reflect')
    return F.pad(x, pad, mode='replicate')


def predict_blocks(fn, x, block, overlap, multiple=POOL, device=None):
    """
    Apply fn to overlapping blocks of the inputs x (a dictionary of tensors (1, C, n1, n2[, n3]) of the same spatial
    shape) and blend the outputs (a dictionary of tensors (1, C', b1, b2[, b3])).

    The blocks start on the pooling grid of the whole image, so a block and the whole image compute the same
    features away from the block edges; the outputs of overlapping blocks are blended with a raised-cosine taper,
    which suppresses the edge effects of every block. With block sizes of 0 (or larger than the image), fn is
    applied to the whole image, padded to a multiple of `multiple`. The inputs and the blended outputs stay on the
    device of x, e.g., in the CPU memory for a large volume, and every block is moved to `device` (if given) for fn.
    """
    key = next(iter(x))
    size = list(x[key].shape[2:])
    axes = [block_starts(n, b, o, multiple) for n, b, o in zip(size, block, overlap)]
    bsize = [a[0] for a in axes]
    total = [a[2] for a in axes]
    ov = [0 if len(a[1]) == 1 else o for a, o in zip(axes, overlap)]
    xp = {k: pad_end(v, total) for k, v in x.items()}

    grid = [[]]
    for a in axes:
        grid = [g + [s] for g in grid for s in a[1]]

    dev = x[key].device
    wb = taper_weight(bsize, ov, dev)
    acc = {}
    wsum = torch.zeros((1, 1, *total), device=dev)
    for start in grid:
        sl = (slice(None), slice(None)) + tuple(slice(s, s + b) for s, b in zip(start, bsize))
        out = fn({k: v[sl] if device is None else v[sl].to(device) for k, v in xp.items()})
        for k, v in out.items():
            if k not in acc:
                acc[k] = torch.zeros((1, v.shape[1], *total), device=dev, dtype=torch.float32)
            acc[k][sl] += wb * v.float().to(dev)
        wsum[sl] += wb

    crop = (slice(None), slice(None)) + tuple(slice(0, n) for n in size)
    return {k: (v / wsum.clamp_min(1.0e-12))[crop] for k, v in acc.items()}

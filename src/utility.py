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

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import matplotlib as mplt
from matplotlib import rcParams


def set_font():
    """Arial for the text and the mathematics of matplotlib figures."""

    basefamily = 'sans-serif'
    basefont = 'Arial'
    rcParams['font.family'] = basefamily
    rcParams['font.' + basefamily] = basefont
    mplt.rcParams['mathtext.fontset'] = 'custom'
    mplt.rcParams['mathtext.rm'] = basefont
    mplt.rcParams['mathtext.sf'] = basefont
    mplt.rcParams['mathtext.it'] = basefont + ':italic'
    mplt.rcParams['mathtext.bf'] = basefont + ':bold'


def str2bool(v):
    """Convert a command-line string to a bool."""

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'y', 'true', 't', 'on', '1'):
        return True
    if v.lower() in ('no', 'n', 'false', 'f', 'off', '0'):
        return False
    raise argparse.ArgumentTypeError('expected one of yes/no, y/n, true/false, t/f, on/off, 1/0')


def read_array(filename, shape, dtype=np.float32, totorch=True):
    """
    Read a raw binary array of the given shape, stored with the first axis fastest (as written by write_array);
    the file size must match the shape.
    """

    count = int(np.prod(shape))
    size = os.path.getsize(filename)
    if size != count * np.dtype(dtype).itemsize:
        raise ValueError('%s has %d bytes, but the shape %s needs %d' % (filename, size, tuple(shape),
                                                                        count * np.dtype(dtype).itemsize))
    x = np.fromfile(filename, count=count, dtype=dtype)
    x = np.transpose(np.reshape(x, shape[::-1]))

    if totorch:
        x = torch.from_numpy(x).type(torch.FloatTensor)

    return x


def write_array(x, filename, dtype=np.float32):
    """Write an array as raw binary data with the first axis fastest."""

    np.transpose(np.asarray(x, dtype=dtype)).tofile(filename)


def set_random_seed(seed=12345):
    """Seed Python, NumPy and PyTorch, and use deterministic convolution algorithms."""

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_numpy(w):
    """Squeezed NumPy array of a tensor."""

    return w.squeeze().data.cpu().numpy()


def date_time():
    """Time stamp for log messages."""

    return datetime.now().strftime(" %Y/%m/%d %H:%M:%S ")

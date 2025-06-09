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


import numpy as np
import os
import torch
from datetime import datetime

import matplotlib as mplt
from matplotlib import rcParams

def set_font():

    basefamily = 'sans-serif'
    basefont = 'Arial'
    fontset = 'custom'
    rcParams['font.family'] = basefamily
    rcParams['font.' + basefamily] = basefont
    mplt.rcParams['mathtext.fontset'] = fontset
    mplt.rcParams['mathtext.rm'] = basefont
    mplt.rcParams['mathtext.sf'] = basefont
    mplt.rcParams['mathtext.it'] = basefont + ':italic'
    mplt.rcParams['mathtext.bf'] = basefont + ':bold'

    return None


## Convert string to bool
def str2bool(v):

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'y', 'true', 't', 'on', '1'):
        return True
    elif v.lower() in ('no', 'n', 'false', 'f', 'off', '0'):
        return False
    else:
        print(' Warning: Argument must be one of yes/no, y/n, true/false, t/f, on/off, 1/0. ')
        exit(0)


## Read a raw binary array and convert to PyTorch tensor
def read_array(filename, shape, dtype=np.float32, totorch=True):

    x = np.fromfile(filename, count=np.prod(shape), dtype=dtype)
    x = np.reshape(x, shape[::-1])
    x = np.transpose(x)

    if totorch:
        x = torch.from_numpy(x).type(torch.FloatTensor)

    return x


## Write a binary array
def write_array(x, filename, dtype=np.float32):

    x = np.asarray(x, dtype=dtype)
    x = np.transpose(x)
    x.tofile(filename)


## Forward integer indices
def forward_range(start, n, step=1):

    r = np.zeros(n, dtype=np.int32)
    for k in range(n):
        r[k] = start + k * step

    return r


##  Set random seeds for training
def set_random_seed(seed=12345):

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    return None


## Transfer data to CPU and extract core array with numpy
def get_numpy(w):

    return w.squeeze().data.cpu().numpy()


## Get time stamp
def date_time():

    now = datetime.now()
    return now.strftime(" %Y/%m/%d %H:%M:%S ")


## Roll over a block by overlapping
def rollover_block(n, b, p, allow_longer=False):

    start = []
    end = []
    ibeg = 1
    iend = b
    start.append(ibeg)
    end.append(iend)
    nb = 1
    
    if allow_longer:
        # if the last chunck can be longer than the max length
        while iend < n:

            ibeg = ibeg + b - p
            iend = iend + b - p
            
            stop = False
            if ibeg > n:
                break
            
            start.append(ibeg)
            end.append(iend)
            nb = nb + 1
        
    else:
        # otherwise, turn back at the tail

        if iend >= n:
            end[0] = n
        else:
            while ibeg <= n:

                ibeg = ibeg + b - p
                iend = iend + b - p

                stop = False
                if iend >= n:
                    iend = n
                    ibeg = iend - b + 1
                    stop = True

                start.append(ibeg)
                end.append(iend)
                nb = nb + 1

                if stop:
                    break

    for i in range(nb):
        start[i] = start[i] - 1
        end[i] = end[i] - 1

    return nb, start, end

# Tapering 2D numpy or pytorch tensors without batch or channel dimension
def taper_2d(x, width, apply=[True, True, True, True]):

    n1, n2 = x.shape

    if apply[0]:
        for i in range(0, width[0]):
            x[i, :] = x[i, :] * i * 1.0 / width[0]
    if apply[1]:
        for i in range(0, width[1]):
            x[n1 - 1 - i, :] = x[n1 - 1 - i, :] * i * 1.0 / width[1]

    if apply[2]:
        for i in range(0, width[2]):
            x[:, i] = x[:, i] * i * 1.0 / width[2]
    if apply[3]:
        for i in range(0, width[3]):
            x[:, n2 - 1 - i] = x[:, n2 - 1 - i] * i * 1.0 / width[3]

    return x


# Tapering 3D numpy or pytorch tensors without batch or channel dimension
def taper_3d(x, width, apply=[True, True, True, True, True, True]):

    n1, n2, n3 = x.shape

    if apply[0]:
        for i in range(0, width[0]):
            x[i, :, :] = x[i, :, :] * i * 1.0 / width[0]
    if apply[1]:
        for i in range(0, width[1]):
            x[n1 - 1 - i, :, :] = x[n1 - 1 - i, :, :] * i * 1.0 / width[1]

    if apply[2]:
        for i in range(0, width[2]):
            x[:, i, :] = x[:, i, :] * i * 1.0 / width[2]
    if apply[3]:
        for i in range(0, width[3]):
            x[:, n2 - 1 - i, :] = x[:, n2 - 1 - i, :] * i * 1.0 / width[3]

    if apply[4]:
        for i in range(0, width[4]):
            x[:, :, i] = x[:, :, i] * i * 1.0 / width[4]
    if apply[5]:
        for i in range(0, width[5]):
            x[:, :, n3 - 1 - i] = x[:, :, n3 - 1 - i] * i * 1.0 / width[5]

    return x


# Tapering 2D numpy or pytorch tensors with batch and channel dimensions
def taper_2d_bc(x, width, apply=[True, True, True, True], method='linear'):

    _, _, n1, n2 = x.shape
    
    if method == 'linear':

        if apply[0]:
            for i in range(0, width[0]):
                x[:, :, i, :] = x[:, :, i, :] * i * 1.0 / width[0]
        if apply[1]:
            for i in range(0, width[1]):
                x[:, :, n1 - 1 - i, :] = x[:, :, n1 - 1 - i, :] * i * 1.0 / width[1]
    
        if apply[2]:
            for i in range(0, width[2]):
                x[:, :, :, i] = x[:, :, :, i] * i * 1.0 / width[2]
        if apply[3]:
            for i in range(0, width[3]):
                x[:, :, :, n2 - 1 - i] = x[:, :, :, n2 - 1 - i] * i * 1.0 / width[3]
                
    elif method == 'zero':
        
        if apply[0]:
            for i in range(0, width[0]):
                x[:, :, i, :] = 0
        if apply[1]:
            for i in range(0, width[1]):
                x[:, :, n1 - 1 - i, :] = 0

        if apply[2]:
            for i in range(0, width[2]):
                x[:, :, :, i] = 0
        if apply[3]:
            for i in range(0, width[3]):
                x[:, :, :, n2 - 1 - i] = 0

    return x


# Tapering 3D numpy or pytorch tensors with batch and channel dimensions
def taper_3d_bc(x, width, apply=[True, True, True, True, True, True], method='linear'):

    _, _, n1, n2, n3 = x.shape
    
    if method == 'linear':

        if apply[0]:
            for i in range(0, width[0]):
                x[:, :, i, :, :] = x[:, :, i, :, :] * i * 1.0 / width[0]
        if apply[1]:
            for i in range(0, width[1]):
                x[:, :, n1 - 1 - i, :, :] = x[:, :, n1 - 1 - i, :, :] * i * 1.0 / width[1]
    
        if apply[2]:
            for i in range(0, width[2]):
                x[:, :, :, i, :] = x[:, :, :, i, :] * i * 1.0 / width[2]
        if apply[3]:
            for i in range(0, width[3]):
                x[:, :, :, n2 - 1 - i, :] = x[:, :, :, n2 - 1 - i, :] * i * 1.0 / width[3]
    
        if apply[4]:
            for i in range(0, width[4]):
                x[:, :, :, :, i] = x[:, :, :, :, i] * i * 1.0 / width[4]
        if apply[5]:
            for i in range(0, width[5]):
                x[:, :, :, :, n3 - 1 - i] = x[:, :, :, :, n3 - 1 - i] * i * 1.0 / width[5]
                
    elif method == 'zero':
        
        if apply[0]:
            for i in range(0, width[0]):
                x[:, :, i, :, :] = 0
            for i in range(0, width[1]):
                x[:, :, n1 - 1 - i, :, :] = 0

        if apply[2]:
            for i in range(0, width[2]):
                x[:, :, :, i, :] = 0
        if apply[3]:
            for i in range(0, width[3]):
                x[:, :, :, n2 - 1 - i, :] = 0

        if apply[4]:
            for i in range(0, width[4]):
                x[:, :, :, :, i] = 0
        if apply[5]:
            for i in range(0, width[5]):
                x[:, :, :, :, n3 - 1 - i] = 0

    return x

# Merge 3D pytorch tensors with batch and channel dimensions
def merge_block_2d(a, b, range, mode='taper', taperlen=(0, 0, 0, 0), rmsnorm=False):
    
    i1, h1, i2, h2 = range
    _, _, n1, n2 = a.shape
        
    if rmsnorm:
        nm = torch.norm(b)
        if nm != 0:
            b = b/nm
        
    if mode == 'add':
        a[:, :, i1:h1, i2:h2] = a[:, :, i1:h1, i2:h2] + b
        
    elif mode == 'taper':
        t1a = False if i1 == 0 else True
        t1b = False if h1 == n1 else True
        t2a = False if i2 == 0 else True
        t2b = False if h2 == n2 else True
        a[:, :, i1:h1, i2:h2] = a[:, :, i1:h1, i2:h2] + taper_2d_bc(b, taperlen, apply=[t1a, t1b, t2a, t2b])
        
    elif mode == 'max':
        a[:, :, i1:h1, i2:h2] = torch.where(a[:, :, i1:h1, i2:h2] > b, a[:, :, i1:h1, i2:h2], b)
        
    elif mode == 'min':
        a[:, :, i1:h1, i2:h2] = torch.where(a[:, :, i1:h1, i2:h2] < b, a[:, :, i1:h1, i2:h2], b)
        
    elif mode == 'signed_max':
        a[:, :, i1:h1, i2:h2] = torch.where(torch.abs(a[:, :, i1:h1, i2:h2]) > torch.abs(b), a[:, :, i1:h1, i2:h2], b)
        
    elif mode == 'signed_min':
        a[:, :, i1:h1, i2:h2] = torch.where(torch.abs(a[:, :, i1:h1, i2:h2]) < torch.abs(b), a[:, :, i1:h1, i2:h2], b)
        
    if mode == 'mean':
        a[:, :, i1:h1, i2:h2] = 0.5*(a[:, :, i1:h1, i2:h2] + b)
        
    return a

    
# Merge 3D pytorch tensors with batch and channel dimensions
def merge_block_3d(a, b, range, mode='taper', taperlen=(0, 0, 0, 0, 0, 0), rmsnorm=False):
    
    i1, h1, i2, h2, i3, h3 = range
    _, _, n1, n2, n3 = a.shape
    
    if rmsnorm:
        nm = torch.norm(b)
        if nm != 0:
            b = b/nm
        
    if mode == 'add':
        a[:, :, i1:h1, i2:h2, i3:h3] = a[:, :, i1:h1, i2:h2, i3:h3] + b
        
    elif mode == 'taper':
        t1a = False if i1 == 0 else True
        t1b = False if h1 == n1 else True
        t2a = False if i2 == 0 else True
        t2b = False if h2 == n2 else True
        t3a = False if i3 == 0 else True
        t3b = False if h3 == n3 else True
        a[:, :, i1:h1, i2:h2, i3:h3] = a[:, :, i1:h1, i2:h2, i3:h3] + taper_3d_bc(b, taperlen, apply=[t1a, t1b, t2a, t2b, t3a, t3b])
        
    elif mode == 'max':
        a[:, :, i1:h1, i2:h2, i3:h3] = torch.where(a[:, :, i1:h1, i2:h2, i3:h3] > b, a[:, :, i1:h1, i2:h2, i3:h3], b)
        
    elif mode == 'min':
        a[:, :, i1:h1, i2:h2, i3:h3] = torch.where(a[:, :, i1:h1, i2:h2, i3:h3] < b, a[:, :, i1:h1, i2:h2, i3:h3], b)
        
    elif mode == 'signed_max':
        a[:, :, i1:h1, i2:h2, i3:h3] = torch.where(torch.abs(a[:, :, i1:h1, i2:h2, i3:h3]) > torch.abs(b), a[:, :, i1:h1, i2:h2, i3:h3], b)
        
    elif mode == 'signed_min':
        a[:, :, i1:h1, i2:h2, i3:h3] = torch.where(torch.abs(a[:, :, i1:h1, i2:h2, i3:h3]) < torch.abs(b), a[:, :, i1:h1, i2:h2, i3:h3], b)

    return a


## Slice view of a 3D volume
import pyvista as pv

"""
Adds three orthogonal slices (slice1, slice2, slice3) of a 3D numpy array to a specified subplot.

Parameters:
    plotter (pyvista.Plotter): The PyVista Plotter instance to which the slices will be added.
    data (numpy.ndarray): The 3D numpy array to visualize.
    colormap (str): The colormap to use for the slices (default: "viridis").
    slice1 (int): Index for the slice along the first dimension (x1). Defaults to the middle.
    slice2 (int): Index for the slice along the second dimension (x2). Defaults to the middle.
    slice3 (int): Index for the slice along the third dimension (x3). Defaults to the middle.
    vmin (float): Minimum value for the color scale. Defaults to the data minimum.
    vmax (float): Maximum value for the color scale. Defaults to the data maximum.
    subplot_index (tuple): The (row, column) index for the subplot.
"""
def add_slices_to_plotter(plotter, data, colormap="viridis", opacity = 1,
                          slice1=None, slice2=None, slice3=None,
                          vmin=None, vmax=None, subplot_index=(0, 0), 
                          title=None, view_up=(-1, 0, 0), colorbar=False, 
                          compass=False, zoom=1.0):

    # Validate input
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D numpy array.")
        
    n1, n2, n3 = data.shape

    # Set default slice indices to the middle if not provided
    slice1 = slice1 if slice1 is not None else n1 // 2
    slice2 = slice2 if slice2 is not None else n2 // 2
    slice3 = slice3 if slice3 is not None else n3 // 2

    # Set default color range
    vmin = vmin if vmin is not None else data.min()
    vmax = vmax if vmax is not None else data.max()
    
    # Grid
    grid = pv.ImageData()
    grid.dimensions = (n1, n2, n3)
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    
    # Assign scalar data
    # For subplots, pyvista only allows scalar data with unique names
    i = subplot_index[0]
    j = subplot_index[1]
    grid.point_data[f"scalars_{i}_{j}"] = data.ravel(order='F')

    # Add slices to the specified subplot
    if title is None:
        t = f"Subplot {subplot_index}"
    else:
        t = title 
        
    scalar_bar_args = {
        "title": t,
        "title_font_size": 12,
        "label_font_size": 10,
        "vertical": False
    }
    
    plotter.subplot(subplot_index[0], subplot_index[1])
    # plotter.add_text(t, position="upper_left", font_size=12)
    plotter.add_title(t, font_size=9)

    if compass:
        axes = plotter.add_axes()
        axes.SetXAxisLabelText("X1")
        axes.SetYAxisLabelText("X2")
        axes.SetZAxisLabelText("X3")
        
    slices = grid.slice_orthogonal(x=slice1, y=slice2, z=slice3)
    
    if colorbar:
        plotter.add_mesh(slices, scalars=f"scalars_{i}_{j}",  
                         cmap=colormap, opacity=opacity, clim=[vmin, vmax], 
                         show_scalar_bar=True, scalar_bar_args=scalar_bar_args,
                         ambient=0.6, diffuse=0.6, specular=0.0,
                         interpolate_before_map=False)
    else:
        plotter.add_mesh(slices, scalars=f"scalars_{i}_{j}",  
                         cmap=colormap, opacity=opacity, clim=[vmin, vmax], 
                         show_scalar_bar=False,
                         ambient=0.6, diffuse=0.6, specular=0.0,
                         interpolate_before_map=False)
    


    camera_position = [
        (n1//2 - 2*n1, n2//2 - 2*n2, n3//2 + 2*n3),     # camera_location
        (n1//2, n2//2, n3//2),      # focal_point (center of volume)
        (-1, 0, 0)         # view_up (X down is "up" in display)
    ]
    plotter.camera_position = camera_position
    plotter.camera.zoom(zoom)


## Volume rendering view of a 3D volume
def add_volume_to_plotter(plotter, data, colormap="viridis", opacity=[[0, 1], [0, 1]],
                          vmin=None, vmax=None, subplot_index=(0, 0), 
                          title=None, view_up=(-1, 0, 0), colorbar=False, 
                          compass=False, zoom=1.0):
    
    # Validate input
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D numpy array.")
        
    n1, n2, n3 = data.shape

    # Set default color range
    vmin = vmin if vmin is not None else data.min()
    vmax = vmax if vmax is not None else data.max()
    
    # Create an ImageData object
    volume = pv.ImageData(dimensions=(n1, n2, n3))
    volume.spacing = (1.0, 1.0, 1.0)   # voxel size
    volume.origin = (0.0, 0.0, 0.0)    # origin
        
    # Assign scalar data
    # For subplots, pyvista only allows scalar data with unique names
    i = subplot_index[0]
    j = subplot_index[1]
    volume.point_data[f"scalars_{i}_{j}"] = data.ravel(order='F')

    # Opacity
    opacity = np.interp(np.linspace(vmin, vmax, 256), opacity[0], opacity[1])*255

    # Add slices to the specified subplot
    if title is None:
        t = f"Subplot {subplot_index}"
    else:
        t = title 
        
    scalar_bar_args = {
        "title": t,
        "title_font_size": 12,
        "label_font_size": 10,
        "vertical": False
    }
    
    plotter.subplot(subplot_index[0], subplot_index[1])
    plotter.add_title(t, font_size=9)

    if compass:
        axes = plotter.add_axes()
        axes.SetXAxisLabelText("X1")
        axes.SetYAxisLabelText("X2")
        axes.SetZAxisLabelText("X3")
    
    if colorbar:
        plotter.add_volume(volume, scalars=f"scalars_{i}_{j}", 
                           cmap=colormap, opacity=opacity, shade=False, 
                           clim=(vmin, vmax), 
                           show_scalar_bar=True, scalar_bar_args=scalar_bar_args, 
                           ambient=0.6, diffuse=0.6, specular=0)
    else:
        plotter.add_volume(volume, scalars=f"scalars_{i}_{j}", 
                           cmap=colormap, opacity=opacity, shade=False, 
                           clim=(vmin, vmax), 
                           show_scalar_bar=False, 
                           ambient=0.6, diffuse=0.6, specular=0)

    camera_position = [
        (n1//2 - 2*n1, n2//2 - 2*n2, n3//2 + 2*n3),     # camera_location
        (n1//2, n2//2, n3//2),      # focal_point (center of volume)
        (-1, 0, 0)         # view_up (X down is "up" in display)
    ]
    plotter.camera_position = camera_position
    plotter.camera.zoom(zoom)


def add_slices_and_volume_to_plotter(plotter, data_slice, data_volume, 
                                     colormap_slice="binary", colormap_volume='jet',
                                     slice1=None, slice2=None, slice3=None,
                                     opacity_slice=1,
                                     opacity_volume=[[0, 1], [0, 1]],
                                     vmin_slice=None, vmax_slice=None, 
                                     vmin_volume=None, vmax_volume=None,
                                     subplot_index=(0, 0), 
                                     title=None, view_up=(-1, 0, 0), colorbar=False, 
                                     compass=False, zoom=1.0):
    
    # Validate input
    if data_slice.ndim != 3:
        raise ValueError("Input data (slice) must be a 3D numpy array.")
    
    if data_volume.ndim != 3:
        raise ValueError("Input data (volume) must be a 3D numpy array.")


    # =========================================================================
    # Slice

    n1, n2, n3 = data_slice.shape

    # Set default slice indices to the middle if not provided
    slice1 = slice1 if slice1 is not None else n1 // 2
    slice2 = slice2 if slice2 is not None else n2 // 2
    slice3 = slice3 if slice3 is not None else n3 // 2

    # Set default color range
    vmin = vmin_slice if vmin_slice is not None else data_slice.min()
    vmax = vmax_slice if vmax_slice is not None else data_slice.max()
    
    # Grid
    grid = pv.ImageData()
    grid.dimensions = (n1, n2, n3)
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    
    # Assign scalar data
    # For subplots, pyvista only allows scalar data with unique names
    i = subplot_index[0]
    j = subplot_index[1]
    grid.point_data[f"scalars_{i}_{j}"] = data_slice.ravel(order='F')

    # Add slices to the specified subplot
    if title is None:
        t = f"Subplot {subplot_index}"
    else:
        t = title 
        
    scalar_bar_args = {
        "title": t,
        "title_font_size": 12,
        "label_font_size": 10,
        "vertical": False
    }
    
    plotter.subplot(subplot_index[0], subplot_index[1])
    plotter.add_title(t, font_size=9)

    if compass:
        axes = plotter.add_axes()
        axes.SetXAxisLabelText("X1")
        axes.SetYAxisLabelText("X2")
        axes.SetZAxisLabelText("X3")
        
    slices = grid.slice_orthogonal(x=slice1, y=slice2, z=slice3)
    
    if colorbar:
        plotter.add_mesh(slices, scalars=f"scalars_{i}_{j}",  
                         cmap=colormap_slice, opacity=opacity_slice, clim=[vmin, vmax], 
                         show_scalar_bar=True, scalar_bar_args=scalar_bar_args,
                         ambient=0.6, diffuse=0.6, specular=0.0,
                         interpolate_before_map=False)
    else:
        plotter.add_mesh(slices, scalars=f"scalars_{i}_{j}",  
                         cmap=colormap_slice, opacity=opacity_slice, clim=[vmin, vmax], 
                         show_scalar_bar=False,
                         ambient=0.6, diffuse=0.6, specular=0.0,
                         interpolate_before_map=False)
    

    # =========================================================================
    # Volume

    n1, n2, n3 = data_volume.shape

    # Set default color range
    vmin = vmin_volume if vmin_volume is not None else data_volume.min()
    vmax = vmax_volume if vmax_volume is not None else data_volume.max()
    
    # Create an ImageData object
    volume = pv.ImageData(dimensions=(n1, n2, n3))
    volume.spacing = (1.0, 1.0, 1.0)   # voxel size
    volume.origin = (0.0, 0.0, 0.0)    # origin
        
    # Assign scalar data
    # For subplots, pyvista only allows scalar data with unique names
    i = subplot_index[0]
    j = subplot_index[1]
    volume.point_data[f"volume_{i}_{j}"] = data_volume.ravel(order='F')

    # Opacity
    opacity = np.interp(np.linspace(vmin, vmax, 256), opacity_volume[0], opacity_volume[1])*255

    # Add slices to the specified subplot
    if title is None:
        t = f"Subplot {subplot_index}"
    else:
        t = title 
        
    scalar_bar_args = {
        "title": t,
        "title_font_size": 12,
        "label_font_size": 10,
        "vertical": False
    }
    
    if compass:
        axes = plotter.add_axes()
        axes.SetXAxisLabelText("X1")
        axes.SetYAxisLabelText("X2")
        axes.SetZAxisLabelText("X3")
    
    if colorbar:
        plotter.add_volume(volume, scalars=f"volume_{i}_{j}", 
                           cmap=colormap_volume, opacity=opacity, shade=False, 
                           clim=(vmin, vmax), 
                           show_scalar_bar=True, scalar_bar_args=scalar_bar_args, 
                           ambient=0.6, diffuse=0.6, specular=0)
    else:
        actor = plotter.add_volume(volume, scalars=f"volume_{i}_{j}", 
                           cmap=colormap_volume, opacity=opacity, shade=False, 
                           clim=(vmin, vmax), 
                           show_scalar_bar=False, 
                           ambient=0.6, diffuse=0.6, specular=0.1, specular_power=20, blending='maximum')

    # =========================================================================
    # Camera

    camera_position = [
        (n1//2 - 2*n1, n2//2 - 2*n2, n3//2 + 2*n3),     # camera_location
        (n1//2, n2//2, n3//2),      # focal_point (center of volume)
        (-1, 0, 0)         # view_up (X down is "up" in display)
    ]
    plotter.camera_position = camera_position
    plotter.camera.zoom(zoom)


"""
    Reads scalar values grouped by unique epochs from multiple TensorBoard log subdirectories.

    Args:
        log_dir (str): Root directory with subdirs like 'version_*'.
        scalar_names (list of str): List of scalar tags, must include 'epoch'.
        reduce (str): Reduction method over steps within an epoch ('last' or 'mean').

    Returns:
        np.ndarray: Array of shape (n_epochs, len(scalar_names)), with epoch and scalar values.
"""
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def read_scalar_by_epoch(log_dir, scalar_names, reduce='last'):
    
    assert 'epoch' in scalar_names, "'epoch' must be one of the scalar_names"

    # Collect all scalar events by tag
    all_events = {name: [] for name in scalar_names}

    for subdir in os.listdir(log_dir):
        full_path = os.path.join(log_dir, subdir)
        if os.path.isdir(full_path) and subdir.startswith("version_"):
            try:
                ea = EventAccumulator(full_path)
                ea.Reload()
                tags = ea.Tags().get('scalars', [])
                for name in scalar_names:
                    if name in tags:
                        all_events[name].extend(ea.Scalars(name))
                    else:
                        print(f"Warning: '{name}' not found in {subdir}")
            except Exception as e:
                print(f"Error reading {subdir}: {e}")

    # Ensure we have some data
    if any(len(events) == 0 for events in all_events.values()):
        print("Incomplete data: some scalar tags are empty.")
        return np.empty((0, len(scalar_names)))

    # All tags should be aligned by step
    # Build a step-to-scalar dictionary
    step_map = defaultdict(dict)
    for name, events in all_events.items():
        for e in events:
            step_map[e.step][name] = e.value

    # Filter steps where all required tags are present
    aligned = [v for k, v in sorted(step_map.items()) if all(n in v for n in scalar_names)]

    # Group by unique epoch values
    epoch_to_data = defaultdict(list)
    for entry in aligned:
        epoch_val = entry['epoch']
        values = [entry[name] for name in scalar_names if name != 'epoch']
        epoch_to_data[epoch_val].append(values)

    # Aggregate per epoch
    result = []
    for epoch, vals in sorted(epoch_to_data.items()):
        arr = np.array(vals)
        if reduce == 'mean':
            reduced = arr.mean(axis=0)
        elif reduce == 'last':
            reduced = arr[-1]
        else:
            raise ValueError("reduce must be 'mean' or 'last'")
        result.append([epoch] + reduced.tolist())

    return np.array(result)


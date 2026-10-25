#
# Figure of the faults of a volume from the image only, from the image and the seismicity, and from the seismicity
# only (the outputs <prefix>.<mode>.fsem of main3.py --mode=all), on three orthogonal slices. Faults (probability at
# least 0.5) are red, and the seismicity is blue on the image and black on white alone.
#
# Usage: python plot_results.py --image=<image> --meq=<seismicity> --prefix=<prefix> --n1=<n1> --n2=<n2> --n3=<n3>
#        --out=<figure> [--slices i1 i2 i3]
#

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ap = argparse.ArgumentParser(description='Faults of a volume from the three input modes of 3D SCF-Net')
ap.add_argument('--image', type=str, required=True, help='image (raw float32, x1 fastest)')
ap.add_argument('--meq', type=str, required=True, help='seismicity location image (raw float32)')
ap.add_argument('--prefix', type=str, required=True, help='prefix of the predictions <prefix>.<mode>.fsem')
ap.add_argument('--n1', type=int, required=True, help='number of samples along x1 (depth)')
ap.add_argument('--n2', type=int, required=True, help='number of samples along x2')
ap.add_argument('--n3', type=int, required=True, help='number of samples along x3')
ap.add_argument('--slices', type=int, nargs=3, default=None, help='indices of the X1, X2 and X3 slices (default: the centers)')
ap.add_argument('--out', type=str, required=True, help='figure file')
args = ap.parse_args()

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 9, 'axes.titlesize': 9,
                     'axes.labelsize': 9})
n1, n2, n3 = args.n1, args.n2, args.n3


def read(f):
    return np.fromfile(f, dtype=np.float32).reshape(n3, n2, n1).T


img = read(args.image)
img = img / max(float(np.sqrt(np.mean(img[img != 0]**2))), 1.0e-30)
meq = np.clip(read(args.meq), 0, 1)
pred = {m: read('%s.%s.fsem' % (args.prefix, m)) for m in ['image', 'both', 'meq']}

i1, i2, i3 = args.slices if args.slices is not None else (n1 // 2, n2 // 2, n3 // 2)
cuts = [(i1, slice(None), slice(None)), (slice(None), i2, slice(None)), (slice(None), slice(None), i3)]
names = ['X1 = %d' % i1, 'X2 = %d' % i2, 'X3 = %d' % i3]
cols = [('Image and Seismicity', 'img', True, None), ('Image Only', 'img', False, pred['image']),
        ('Image + Seismicity', 'img', True, pred['both']), ('Seismicity Only', 'meq', False, pred['meq'])]
red = ListedColormap(['#ff0000'])
blue = ListedColormap(['#0000ff'])
# modes in columns and slices in rows, every row as high as its slices
heights = [n2, n1, n1]
w = 7.2 / len(cols)
fig, ax = plt.subplots(3, len(cols), figsize=(7.2, w * sum(heights) / max(n2, n3) + 0.5), gridspec_kw={'height_ratios': heights},
                       squeeze=False)
for c, (title, back, with_meq, faults) in enumerate(cols):
    for r, sl in enumerate(cuts):
        a = ax[r][c]
        if back == 'meq':
            a.imshow(meq[sl], cmap='binary', vmin=0, vmax=1, interpolation='none', aspect='auto')
        else:
            a.imshow(img[sl], cmap='gray', vmin=-2.5, vmax=2.5, interpolation='none', aspect='auto')
            if with_meq:
                a.imshow(np.ma.masked_where(meq[sl] < 0.5, meq[sl]), cmap=blue, interpolation='none', aspect='auto')
        if faults is not None:
            a.imshow(np.ma.masked_where(faults[sl] < 0.5, faults[sl]), cmap=red, interpolation='none', aspect='auto')
        a.set_xticks([])
        a.set_yticks([])
        if r == 0:
            a.set_title(title)
        if c == 0:
            a.set_ylabel(names[r])
plt.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.3)
plt.savefig(args.out, dpi=200, bbox_inches='tight', pad_inches=0.02)
print('figure', args.out)

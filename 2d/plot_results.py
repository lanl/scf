#
# Figure of the faults of an image from the image only, from the image and the seismicity, and from the seismicity
# only (the outputs <prefix>.<mode>.fsem of main2.py --mode=all). Faults (probability at least 0.5) are red, and the
# seismicity is blue on the image and black on white alone.
#
# Usage: python plot_results.py --image=<image> --meq=<seismicity> --prefix=<prefix> --n1=<n1> --n2=<n2> --out=<figure>
#

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ap = argparse.ArgumentParser(description='Faults of an image from the three input modes of 2D SCF-Net')
ap.add_argument('--image', type=str, required=True, help='image (raw float32, x1 fastest)')
ap.add_argument('--meq', type=str, required=True, help='seismicity location image (raw float32)')
ap.add_argument('--prefix', type=str, required=True, help='prefix of the predictions <prefix>.<mode>.fsem')
ap.add_argument('--n1', type=int, required=True, help='number of samples along x1 (depth)')
ap.add_argument('--n2', type=int, required=True, help='number of samples along x2')
ap.add_argument('--out', type=str, required=True, help='figure file')
args = ap.parse_args()

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 9, 'axes.titlesize': 9})


def read(f):
    return np.fromfile(f, dtype=np.float32).reshape(args.n2, args.n1).T


img = read(args.image)
img = img / max(float(np.sqrt(np.mean(img[img != 0]**2))), 1.0e-30)
meq = np.clip(read(args.meq), 0, 1)
pred = {m: read('%s.%s.fsem' % (args.prefix, m)) for m in ['image', 'both', 'meq']}

panels = [('Image and Seismicity', 'img', True, None), ('Image Only', 'img', False, pred['image']),
          ('Image + Seismicity', 'img', True, pred['both']), ('Seismicity Only', 'meq', False, pred['meq'])]
wide = args.n2 >= 2 * args.n1
if wide:
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 4 * 7.2 * args.n1 / args.n2 + 1.0))
else:
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 2 * 3.6 * args.n1 / args.n2 + 0.6))
red = ListedColormap(['#ff0000'])
blue = ListedColormap(['#0000ff'])
for a, (title, back, with_meq, faults) in zip(np.ravel(ax), panels):
    if back == 'meq':
        a.imshow(meq, cmap='binary', vmin=0, vmax=1, interpolation='none', aspect='equal')
    else:
        a.imshow(img, cmap='gray', vmin=-2.5, vmax=2.5, interpolation='none', aspect='equal')
        if with_meq:
            a.imshow(np.ma.masked_where(meq < 0.5, meq), cmap=blue, interpolation='none', aspect='equal')
    if faults is not None:
        a.imshow(np.ma.masked_where(faults < 0.5, faults), cmap=red, interpolation='none', aspect='equal')
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(title)
plt.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.4)
plt.savefig(args.out, dpi=200, bbox_inches='tight', pad_inches=0.02)
print('figure', args.out)

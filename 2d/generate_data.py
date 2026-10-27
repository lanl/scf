#
# Generate 2D training and validation data for SCF-Net with the RGM Python interface (https://github.com/lanl/rgm).
# Outputs, in raw float32 with x1 (depth) as the fast axis:
#
#   data_*/<i>_img.bin          seismic image, scaled to unit RMS
#   data_*/<i>_meq.bin          seismicity location image (src/seismicity.py, simulate_seismicity)
#   target_*/<i>_fsem.bin       fault mask (0/1)
#   target_*/<i>_fdip.bin       fault dip / 180
#   target_*/<i>_fsem_meq.bin   faults indicated by the seismicity (the label of the seismicity-only mode)
#   target_*/<i>_params.json    parameters of the model, the image and the seismicity
#
# Every model draws a zoom factor s log-uniformly from --zoom: the wavelength, the layer thickness, the fault
# displacement, the lateral smoothness of reflectors, and the location errors of the seismicity are s times their values
# at s = 1, and the number of faults is 1/s times, while the fault labels and the speckle keep their width, so that the
# network does not depend on the sampling of the image. Along the faults, the seismicity is patchy and clustered, with
# inactive faults, location errors and background events, and in some models it is confined to a region, sparse, or in
# chains along the faults (simulate_seismicity).
#
# Parallel generation: with MPI (mpi4py), e.g., mpirun -np 64 python generate_data.py --mpi ..., every process
# generates every 64th model; without MPI, processes can generate ranges of models (--ibeg, --iend). Every model
# depends only on its index and --seed, so the data do not depend on the number of processes.
#
# Note: rgm must be imported before torch (with torch imported first, RGM returns NaN images); this script does not
# import torch.

import rgm
import argparse
import json
import os
import sys
import traceback

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from seismicity import simulate_seismicity

parser = argparse.ArgumentParser(description='Generate 2D SCF-Net data with RGM')
parser.add_argument('--outdir', type=str, default='./dataset', help='output directory')
parser.add_argument('--ntrain', type=int, default=2000, help='number of training models')
parser.add_argument('--nvalid', type=int, default=200, help='number of validation models')
parser.add_argument('--n1', type=int, default=256, help='number of depth samples')
parser.add_argument('--n2', type=int, default=256, help='number of lateral samples')
parser.add_argument('--seed', type=int, default=2026, help='base random seed')
parser.add_argument('--mpi', action='store_true', help='distribute the models over the MPI processes (mpi4py)')
parser.add_argument('--ibeg', type=int, default=0, help='first model index')
parser.add_argument('--iend', type=int, default=-1, help='last model index, exclusive (-1 = all)')
parser.add_argument('--zoom', type=float, nargs=2, default=[1.0, 3.0], help='range of the grid refinement factor, log-uniform')
parser.add_argument('--fs', type=float, nargs=2, default=[130.0, 220.0], help='range of the wavelet frequency at zoom 1 (Hz, dt = 1 ms)')
parser.add_argument('--thick', type=float, nargs=2, default=[0.7, 1.5],
                    help='range of the mean layer thickness relative to the wavelet period, log-uniform')
parser.add_argument('--nf', type=int, nargs=2, default=[1, 12], help='range of the number of faults at zoom 1')
parser.add_argument('--disp_min', type=float, nargs=2, default=[1.0, 3.0], help='range of the smallest fault displacement at zoom 1, samples')
parser.add_argument('--disp_max', type=float, nargs=2, default=[8.0, 30.0], help='range of the largest fault displacement at zoom 1, samples')
parser.add_argument('--relief', type=float, nargs=2, default=[4.0, 60.0], help='range of the reflector relief at zoom 1, samples, log-uniform')
parser.add_argument('--slope', type=float, nargs=2, default=[1.0, 50.0], help='range of the reflector rise across the model, samples, log-uniform')
parser.add_argument('--noise', type=float, nargs=2, default=[0.0, 0.6],
                    help='range of the level of the migration noise and of the random noise convolved with the point spread '
                    'function: the RMS of the noise relative to that of the image')
parser.add_argument('--speckle', type=float, nargs=2, default=[0.0, 1.0],
                    help='range of the level of the speckle (random noise added to the image, as Kirchhoff migration noise), '
                    'which lies mostly outside the band of the image: the RMS of the speckle relative to that of the image')
parser.add_argument('--p_conv_noise', type=float, default=0.8,
                    help='share of the random noise (other than migration noise) convolved with the point spread function, '
                    'with the frequencies and wavenumbers of the image (except with dip-independent imaging); the rest is added '
                    'to the image')
parser.add_argument('--fwidth', type=float, default=2.0, help='width of the fault labels, samples')
parser.add_argument('--p_none', type=float, default=0.05, help='probability of a model without faults')
parser.add_argument('--p_regular', type=float, default=0.35,
                    help='probability of quasi-regular (quasi-parallel) fault sets; the remaining models have random faults')
parser.add_argument('--p_thrust', type=float, default=0.25,
                    help='probability of thrust structures (propagation, bend, imbricate, duplex or pop-up)')
parser.add_argument('--p_high_relief', type=float, default=0.5,
                    help='share of the models without thrusts that have a high reflector relief (--high_relief)')
parser.add_argument('--high_relief', type=float, nargs=2, default=[0.6, 1.0],
                    help='range of the high reflector relief, as a fraction of n1')
parser.add_argument('--p_unconf', type=float, default=0.2, help='probability of unconformities')
parser.add_argument('--p_salt', type=float, default=0.25,
                    help='share of the models with salt bodies (without layers or faults), from their own random stream')
parser.add_argument('--p_vary_disp', type=float, default=0.5, help='probability of faults with a displacement that varies along the fault')
parser.add_argument('--p_short', type=float, default=0.5,
                    help='share of the models with varying displacement whose slip patches are small, which gives short faults')
parser.add_argument('--preset', type=str, default='general', choices=['general', 'opunake'],
                    help='general: the random models above; opunake: models that mimic the Opunake image and its '
                    'seismicity, for fine tuning (use --n2=1024)')
opts = parser.parse_args()

UNCONF_SHAPES = ['meander_channel', 'meander_canyon', 'drainage_channel', 'drainage_canyon']


def write_array(x, filename):
    np.asarray(x, dtype=np.float32).T.tofile(filename)


def loguniform(r, ab):
    a, b = ab
    return float(np.exp(r.uniform(np.log(a), np.log(b))))


def model_rng(i, attempt):
    """Random stream of model i; a new attempt (after a non-finite image) draws a new model."""

    return np.random.default_rng([opts.seed, i, 0] if attempt == 0 else [opts.seed, i, 0, attempt])


def seismicity_rng(i, attempt):
    """Random stream of the seismicity of model i."""

    return np.random.default_rng([opts.seed, i, 1] if attempt == 0 else [opts.seed, i, 1, attempt])


def salt_rng(i, attempt):
    """Random stream of the salt bodies of model i, so that the models without salt do not depend on them."""

    return np.random.default_rng([opts.seed, i, 3] if attempt == 0 else [opts.seed, i, 3, attempt])


def generate(i, attempt=0):
    """Model i: image, labels, seismicity and a description of the model (None for a non-finite image)."""

    n1, n2 = opts.n1, opts.n2
    rng = model_rng(i, attempt)

    # Grid refinement
    s = loguniform(rng, opts.zoom)
    p = rgm.rgm2(n1=n1, n2=n2, seed=int(rng.integers(1, 2**30)))
    meta = {'zoom': s}

    # Wavelet: Ricker with a residual phase rotation, mostly within +-30 degrees
    f0 = rng.uniform(opts.fs[0], opts.fs[1]) / s
    period = 1000.0 / f0
    p.f0 = f0
    p.wave = 'ricker'
    pmax = 90.0 if rng.random() < 0.2 else 30.0
    p.wave_phase = rng.uniform(-pmax, pmax)

    # Layers, with a thickness that varies independently of the wavelength
    nl = max(5, int(round(n1 / (period * loguniform(rng, opts.thick)))))
    p.nl = nl
    # layer thickness varying from layer to layer (lwv), and laterally (lwh, at most about +-20%: larger lateral variations
    # make layers swell and pinch unlike most sedimentary layers)
    lwv, lwh = rng.uniform(0.0, 0.4), rng.uniform(0.0, 0.1)
    p.lwv = lwv
    p.lwh = lwh
    meta.update(lwv=lwv, lwh=lwh)
    p.fwidth = opts.fwidth

    # Faults: none, quasi-regular sets, thrust structures with a few normal faults, or random faults
    r = rng.random()
    if r < opts.p_none:
        family = 'none'
    elif r < opts.p_none + opts.p_regular:
        family = 'regular'
    elif r < opts.p_none + opts.p_regular + opts.p_thrust:
        family = 'thrust'
    else:
        family = 'normal'

    # Reflector shape: thrust models have nearly flat layers, so that the thrust structures are clear; of the other
    # models, a share (--p_high_relief) has a high relief of a large fraction of the model depth (--high_relief)
    if family == 'thrust':
        relief = loguniform(rng, (2.0, 15.0)) * s
    elif rng.random() < opts.p_high_relief:
        relief = rng.uniform(opts.high_relief[0], opts.high_relief[1]) * n1
        meta['high_relief'] = True
    else:
        relief = min(loguniform(rng, opts.relief) * s, 0.3 * n1)
    slope = float(np.copysign(loguniform(rng, opts.slope), rng.uniform(-1.0, 1.0)))
    p.refl_slope = slope
    p.refl_slope_top = slope / 5.0
    p.refl_height = [0.0, relief]
    # bottom reflector: a high relief comes with broad bumps or folds, whose width grows with the relief, so that the
    # steepest dips stay at about 45 to 65 degrees (random or Perlin surfaces of a high relief give contorted, near-vertical
    # layers)
    if meta.get('high_relief'):
        shape = ['gaussian', 'cauchy', 'fold'][int(rng.integers(0, 3))]
        width = relief
    else:
        shape = ['gaussian', 'cauchy', 'perlin', 'random', 'fold', 'fold'][int(rng.integers(0, 6))]
        width = 0.0
    # top reflector: in half of the models, the shape of the bottom reflector with a similar relief, so that the layers stay
    # nearly parallel; otherwise another shape with a gentle relief, between which and the bottom reflector the layers
    # converge and diverge, as growth strata
    if rng.random() < 0.5:
        shape_top = 'same'
        p.refl_height_top = [0.0, relief * rng.uniform(0.6, 1.0)]
    else:
        shape_top = [v for v in ['gaussian', 'cauchy', 'perlin', 'random', 'fold'] if v != shape][int(rng.integers(0, 4))]
        p.refl_height_top = [0.0, min(rng.uniform(2.0, 10.0) * s, relief)]
    p.refl_shape = shape
    p.refl_shape_top = shape_top
    # parameters of the shapes, which the two reflectors share, except the smoothing of random and Perlin surfaces (at most
    # 15 samples, so that the surfaces keep their roughness)
    shapes = {shape, shape_top}
    if shapes & {'gaussian', 'cauchy'}:
        p.refl_mu2 = [0.0, n2 - 1.0]
        p.refl_sigma2 = [max(20.0 * s, 0.25 * width), max(50.0 * s, 0.6 * width)]
        p.ng = int(rng.integers(1, 4))
        if rng.random() < 0.5:
            p.refl_skew = [0.3, 0.7]
            p.refl_skew_common = bool(rng.random() < 0.5)
    if 'fold' in shapes:
        p.refl_fold_lambda = [max(0.2 * n2 * s, 1.5 * width), max(0.5 * n2 * s, 3.0 * width)]
        p.refl_fold_vergence = [0.0, 0.6]
        p.refl_fold_lambda_drift = 0.5
        p.refl_fold_amp_drift = 0.6
    smooth = {'perlin': min(4.0 * s, 15.0), 'random': min(10.0 * s, 15.0)}
    p.refl_smooth = smooth.get(shape, 0.0)
    p.refl_smooth_top = smooth.get(shape_top, 0.0)
    meta.update(f0=f0, nl=nl, shape=shape, shape_top=shape_top, relief=relief, slope=slope)

    disp_min = rng.uniform(opts.disp_min[0], opts.disp_min[1]) * s
    disp_max = max(disp_min, rng.uniform(opts.disp_max[0], opts.disp_max[1]) * s)
    disp_sign = 1.0 if rng.random() < 0.5 else -1.0
    p.delta_dip = [0.0, rng.uniform(0.0, 25.0)]
    vary_disp = bool(rng.random() < opts.p_vary_disp)
    short = vary_disp and bool(rng.random() < opts.p_short)
    p.yn_vary_disp = vary_disp

    fdisp = [0.0, 0.0]
    if family == 'none':
        nf = 0
    elif family == 'regular':
        p.yn_regular_fault = True
        nf = max(2, int(round(rng.integers(8, 21) / s)))
        if rng.random() < 0.5:
            p.dip = [rng.uniform(100.0, 120.0), rng.uniform(60.0, 80.0)]
        else:
            p.dip = [rng.uniform(60.0, 80.0), rng.uniform(100.0, 120.0)]
        fdisp = list(np.where(rng.random(2) < 0.5, 1.0, -1.0) * rng.uniform(3.0, 10.0) * s)
    else:
        nf = max(1, int(round(rng.integers(opts.nf[0], opts.nf[1] + 1 + (4 if short else 0)) / s)))
        fdisp = [disp_min, disp_max] if disp_sign > 0 else [-disp_max, -disp_min]
        p.dip = [35.0, 145.0] if rng.random() < 0.3 else [50.0, 130.0]
        if family == 'thrust':
            structure = ['propagation', 'bend', 'imbricate', 'duplex', 'popup'][int(rng.integers(0, 5))]
            p.thrust_vergence = ['left', 'right'][int(rng.integers(0, 2))]
            p.thrust_dip = [15.0, 40.0]
            if structure == 'propagation':
                p.nthrust = 1
                p.thrust_z = [1.1, 1.2]
                p.thrust_tip_z = [0.2, 0.5]
            elif structure == 'bend':
                p.nthrust = 1
                p.thrust_roof = 1.0
                p.thrust_z = [0.85, 0.95]
                p.thrust_roof_z = [0.3, 0.5]
                p.thrust_disp = [0.15 * n2, 0.25 * n2]
            elif structure == 'imbricate':
                p.nthrust = int(rng.integers(2, 5))
                p.thrust_z = [1.0, 1.2]
                p.thrust_tip_z = [0.0, 0.5]
            elif structure == 'duplex':
                p.nthrust = int(rng.integers(3, 5))
                p.thrust_roof = 1.0
                p.thrust_z = [0.85, 0.9]
                p.thrust_roof_z = [0.45, 0.55]
            else:
                p.nthrust = 2
                p.thrust_back = 1.0
                p.thrust_tip_z = [0.2, 0.45]
            nf = int(round(rng.integers(0, 3) / s))
            meta['thrust'] = structure
    p.nf = nf
    p.disp = fdisp
    if vary_disp:
        # finite slip patches (small ones give short faults): RGM tapers the displacement with the depth of every point of
        # the displaced block, which kinks the layers along the patch boundary, across the block, by a strain of about
        # 2 x displacement / patch half-height; the half-height is at least 6 times the largest displacement (small faults
        # have small displacements), and the displacement decays away from the fault, as the deformation halo of a finite
        # slip patch, over at least 3 times the largest displacement, which confines the kinks to the fault tips
        dmax = max(abs(float(v)) for v in fdisp)
        r = max(0.1 if short else 0.6, 6.0 * dmax / n1)
        p.disp_radius_dip = [r, max(0.4 if short else 1.2, r)]
        r = max(0.1, 3.0 * dmax / n1)
        p.yn_disp_decay = True
        p.disp_decay_width = [r, max(0.3, r)]
    meta.update(family=family, nf=nf, disp=[float(v) for v in fdisp], vary_disp=vary_disp, short=short)

    # Unconformities, Perlin-noise or geomorphological surfaces
    if rng.random() < opts.p_unconf:
        unconf = int(rng.integers(1, 3))
        p.unconf = unconf
        p.unconf_z = [0.05, 0.35]
        p.unconf_height = [5.0 * s, 15.0 * s]
        if rng.random() < 0.6:
            p.unconf_shape = UNCONF_SHAPES[int(rng.integers(0, len(UNCONF_SHAPES)))]
        meta['unconf'] = unconf

    # Salt bodies, from their own random stream, in a share of the models (--p_salt): one to three bodies with a smooth
    # top and radius, of sizes that scale with the zoom; RGM removes the layers and the fault labels inside the salt
    srng = salt_rng(i, attempt)
    if srng.random() < opts.p_salt:
        nsalt = int(srng.integers(1, 4))
        p.yn_salt = True
        p.nsalt = nsalt
        p.salt_radius = [min(0.06 * n2 * s, 0.3 * n2), min(0.16 * n2 * s, 0.3 * n2)]
        p.salt_top_z = [0.3, 0.7]
        p.salt_top_height = min(srng.uniform(0.01, 0.05) * n1 * s, 0.3 * n1)
        p.salt_radius_variation = srng.uniform(0.3, 0.7)
        p.salt_path_variation = srng.uniform(0.0, 0.01) * n2 * s
        meta['salt'] = nsalt

    # Imaging: point spread function, noise and distortions of migrated images
    p.psf_sigma = [12.0 * s, rng.uniform(1.0, 3.0) * s]
    # dip-independent imaging (isotropic point spread function), mostly for thrust structures, whose low-angle thrusts,
    # ramps and steep limbs it images like reflectors, and for high reliefs, whose steep flanks it images; otherwise in
    # half of the models, since dip-dependent imaging leaves steep layers and the faults in them blank
    dip_independent = bool(rng.random() < (0.8 if family == 'thrust' else (0.6 if meta.get('high_relief') else 0.5)))
    p.yn_dip_independent = dip_independent
    # noise level: the RMS of the noise relative to that of the image, drawn as a fraction of the range of the noise type
    level = rng.random()
    if rng.random() < 0.4:
        noise = 'migration'
        noise_level = opts.noise[0] + level * (opts.noise[1] - opts.noise[0])
    else:
        noise = ['normal', 'uniform'][int(rng.integers(0, 2))]
        # random noise, mostly convolved with the point spread function, so that it has the frequencies and wavenumbers of the
        # image (the hardest case for fault detection); otherwise speckle added to the image, white or smoothed by at most
        # half a sample at any zoom, as Kirchhoff migration noise (smoother noise looks like blobs, unlike seismic noise)
        # (not convolved with dip-independent imaging, whose isotropic point spread function would turn the noise into blobs)
        conv_noise = bool(rng.random() < opts.p_conv_noise) and not dip_independent
        noise_range = opts.noise if conv_noise else opts.speckle
        noise_level = noise_range[0] + level * (noise_range[1] - noise_range[0])
        ns = [0.0] * 2 if conv_noise else [float(v) for v in rng.uniform(0.0, 0.5, size=2)]
        p.noise_smooth = ns
        p.yn_conv_noise = conv_noise
        meta.update(noise_smooth=ns, conv_noise=conv_noise)
    p.noise_type = noise
    if rng.random() < 0.25:
        p.f0_bottom = f0 * rng.uniform(0.6, 0.9)
    if rng.random() < 0.5:
        p.illum_level = rng.uniform(0.0, 0.35)
        p.illum_smooth = 30.0 * s
        p.jitter_shift = [rng.uniform(0.0, 1.0), rng.uniform(0.0, 0.3)]
        p.jitter_gain = rng.uniform(0.0, 0.1)
    meta.update(noise=noise, noise_level=noise_level, dip_independent=dip_independent)

    # RGM scales the migration noise to the RMS of the image, but random noise to the peak of the image (or of the
    # reflectivity), which makes the signal-to-noise ratio depend on the model; random noise comes instead from a second
    # run with the same random streams, as the difference from the image without noise, scaled to the RMS of the image
    p.noise_level = noise_level if noise == 'migration' else 0.0
    p.generate()
    img = np.array(p.image)
    if noise != 'migration' and noise_level > 0:
        p.noise_level = 1.0
        p.generate()
        nz = np.array(p.image) - img
        img = img + nz * (noise_level * float(np.sqrt(np.mean(img**2))) / max(float(np.sqrt(np.mean(nz**2))), 1.0e-30))
    try:
        fault = np.array(p.fault)
        fdip = np.array(p.fault_dip) / 180.0
    except AttributeError:
        fault = np.zeros((n1, n2), dtype=np.float32)
        fdip = np.zeros((n1, n2), dtype=np.float32)
    p.free()

    if not np.all(np.isfinite(img)):
        return None
    img = img / max(float(np.sqrt(np.mean(img**2))), 1.0e-30)

    # Seismicity, from its own random stream
    sim = simulate_seismicity(fault, seismicity_rng(i, attempt), scale=s)
    meta['seismicity'] = sim['info']

    return img, fault, fdip, sim, meta


def generate_opunake(i, attempt=0):
    """
    Model i that mimics the Opunake image (as the published fine-tuning data): one broad anticline, 15 to 30 straight
    or curved faults, low noise, a strong smooth gain, and seismicity in chains within a few samples of about two
    thirds of the faults (None for a non-finite image).
    """

    n1, n2 = opts.n1, opts.n2
    rng = model_rng(i, attempt)
    p = rgm.rgm2(n1=n1, n2=n2, seed=int(rng.integers(1, 2**30)))

    f0 = rng.uniform(130.0, 170.0)
    p.f0 = f0
    p.nl = int(round(50.0 + (f0 - 130.0) / 40.0 * 20.0))
    p.fwidth = opts.fwidth
    p.wave = 'ricker'
    p.lwv = 0.25
    p.refl_shape = 'gaussian'
    p.refl_mu2 = [0.1 * n2, 0.9 * n2]
    p.refl_sigma2 = [150.0, 225.0]
    p.ng = 1
    slope = rng.uniform(-25.0, 25.0)
    p.refl_slope = slope
    p.refl_height = [0.0, 75.0]
    p.refl_slope_top = slope / 5.0
    p.refl_smooth_top = 20.0
    p.refl_height_top = [0.0, 5.0]

    if rng.random() < 1.0 / 3.0:
        p.unconf = int(rng.integers(1, 3))
        p.unconf_z = [0.01, 0.15]

    w = n2 / 1024.0
    short = bool(rng.random() < 0.5)
    nf = int(rng.integers(20, 31)) if short else int(rng.integers(15, 26))
    nf = max(2, int(round(nf * w)))
    p.nf = nf
    if short:
        # faults that die out within the model, as in the complex central part of the Opunake image
        p.yn_vary_disp = True
        p.disp_radius_dip = [0.15, 0.5]
    if nf > 21 * w and not short:
        p.yn_regular_fault = True
        if rng.random() < 0.5:
            p.dip = [rng.uniform(100.0, 120.0), rng.uniform(60.0, 80.0)]
            p.disp = [4.0, -4.0]
        else:
            p.dip = [rng.uniform(60.0, 80.0), rng.uniform(100.0, 120.0)]
            p.disp = [-4.0, 4.0]
    else:
        p.disp = [4.0, 10.0]
        p.dip = [50.0, 130.0]
    p.delta_dip = [0.0, 5.0] if rng.random() < 0.5 else [0.0, 35.0]

    ns = rng.uniform(0.0, 1.5, size=2)
    ns[ns < 0.5] = 0.0
    p.noise_smooth = [float(ns[0]), float(ns[1])]
    p.noise_level = rng.uniform(0.0, 0.1)
    p.psf_sigma = [15.0, 5.0]
    p.generate()
    img = np.array(p.image)
    fault = np.array(p.fault)
    fdip = np.array(p.fault_dip) / 180.0
    p.free()
    if not np.all(np.isfinite(img)):
        return None

    # strong smooth gain, as in the published fine-tuning data
    g = ndimage.gaussian_filter(rng.standard_normal((n1, n2)), rng.uniform(4.0, 14.0))
    g = 0.1 + 0.9 * (g - g.min()) / max(g.max() - g.min(), 1.0e-12)
    img = img * g
    img = (img - img.mean()) / max(float(img.std()), 1.0e-30)

    sim = simulate_seismicity(fault, seismicity_rng(i, attempt), scale=1.0, p_scenario=(0.1, 0.1, 0.0, 0.8),
                              chain_density=(0.06, 0.16), chain_offset=(5.0, 12.0), chain_background=0.1, chain_active=(0.6, 0.75))
    meta = {'preset': opts.preset, 'zoom': 1.0, 'family': 'opunake', 'nf': nf, 'short': short, 'f0': f0, 'seismicity': sim['info']}

    return img, fault, fdip, sim, meta


def generate_checked(i, attempts=10):
    """Model i, with new random models in place of models with a non-finite image."""

    for attempt in range(attempts):
        out = (generate_opunake if opts.preset == 'opunake' else generate)(i, attempt)
        if out is not None:
            return out
        print('model %d: non-finite image, new random model' % i, flush=True)
    raise RuntimeError('model %d: non-finite image after %d attempts' % (i, attempts))


if __name__ == '__main__':

    # Processes started by an MPI launcher (Open MPI, MPICH, Intel MPI)
    launched = max(int(os.environ.get(v, '1')) for v in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE'))
    if opts.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
        if launched > 1 and size != launched:
            raise SystemExit('mpi4py uses another MPI library than the launcher (%d instead of %d processes)' % (size, launched))
    else:
        if launched > 1:
            raise SystemExit('started by an MPI launcher with %d processes: add --mpi to distribute the models' % launched)
        comm, rank, size = None, 0, 1

    nm = opts.ntrain + opts.nvalid
    iend = nm if opts.iend < 0 else min(opts.iend, nm)
    if rank == 0:
        print('models %d to %d in %d processes' % (opts.ibeg, iend - 1, size), flush=True)
    for sub in ('data_train', 'data_valid', 'target_train', 'target_valid'):
        os.makedirs(os.path.join(opts.outdir, sub), exist_ok=True)

    try:
        for i in range(opts.ibeg + rank, iend, size):

            img, fault, fdip, sim, meta = generate_checked(i)

            j, split = (i, 'train') if i < opts.ntrain else (i - opts.ntrain, 'valid')
            d = '%s/data_%s/%d' % (opts.outdir, split, j)
            t = '%s/target_%s/%d' % (opts.outdir, split, j)
            write_array(img, d + '_img.bin')
            write_array(sim['meq'], d + '_meq.bin')
            write_array(fault > 0, t + '_fsem.bin')
            write_array(fdip * (fault > 0), t + '_fdip.bin')
            write_array(sim['fsem_meq'], t + '_fsem_meq.bin')
            with open(t + '_params.json', 'w') as f:
                json.dump(meta, f)

            print('%s %d zoom %.2f family %s nf %d seismicity %s events %d' % (split, j, meta['zoom'], meta['family'],
                  meta['nf'], sim['info']['scenario'], sim['info']['nevent']), flush=True)
    except Exception:
        # a failed process ends all processes, so that the job does not wait for it
        traceback.print_exc()
        sys.stdout.flush()
        if size > 1:
            comm.Abort(1)
        raise

    if comm is not None:
        comm.Barrier()
    if rank == 0:
        print('models %d to %d written to %s' % (opts.ibeg, iend - 1, opts.outdir), flush=True)

#
# Generate 3D training and validation data for SCF-Net with the RGM Python interface (https://github.com/lanl/rgm).
# Outputs, in raw float32 with x1 (depth) as the fastest axis:
#
#   data_*/<i>_img.bin          seismic image, scaled to unit RMS
#   data_*/<i>_meq.bin          seismicity location image (src/seismicity.py, simulate_seismicity)
#   target_*/<i>_fsem.bin       fault mask (0/1)
#   target_*/<i>_fdip.bin       fault dip / 180
#   target_*/<i>_fstrike.bin    fault strike / 180
#   target_*/<i>_fsem_meq.bin   faults indicated by the seismicity (the label of the seismicity-only mode)
#   target_*/<i>_params.json    parameters of the model, the image and the seismicity
#
# As in 2d/generate_data.py, every model has a zoom factor s between 1 and 3 that scales the wavelength, the layer
# thickness, the displacement and the lateral smoothness, and the seismicity follows one of four scenarios (full,
# local, sparse, chain), with tens to thousands of events per volume.
#
# Parallel generation: with MPI (mpi4py), e.g., mpirun -np 64 python generate_data.py --mpi ..., every process
# generates every 64th model; without MPI, processes can generate ranges of models (--ibeg, --iend). Every model
# depends only on its index and --seed, so the data do not depend on the number of processes. Use OMP_NUM_THREADS=1 or 2 per
# process.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from seismicity import simulate_seismicity

parser = argparse.ArgumentParser(description='Generate 3D SCF-Net data with RGM')
parser.add_argument('--outdir', type=str, default='./dataset', help='output directory')
parser.add_argument('--ntrain', type=int, default=1000, help='number of training models')
parser.add_argument('--nvalid', type=int, default=100, help='number of validation models')
parser.add_argument('--n1', type=int, default=128, help='number of depth samples')
parser.add_argument('--n2', type=int, default=128, help='number of samples along x2')
parser.add_argument('--n3', type=int, default=128, help='number of samples along x3')
parser.add_argument('--seed', type=int, default=2027, help='base random seed')
parser.add_argument('--mpi', action='store_true', help='distribute the models over the MPI processes (mpi4py)')
parser.add_argument('--ibeg', type=int, default=0, help='first model index')
parser.add_argument('--iend', type=int, default=-1, help='last model index, exclusive (-1 = all)')
parser.add_argument('--zoom', type=float, nargs=2, default=[1.0, 3.0], help='range of the grid refinement factor, log-uniform')
parser.add_argument('--fs', type=float, nargs=2, default=[130.0, 220.0], help='range of the wavelet frequency at zoom 1 (Hz, dt = 1 ms)')
parser.add_argument('--thick', type=float, nargs=2, default=[0.7, 1.5],
                    help='range of the mean layer thickness relative to the wavelet period, log-uniform')
parser.add_argument('--nf', type=int, nargs=2, default=[2, 12], help='range of the number of random faults at zoom 1')
parser.add_argument('--disp_min', type=float, nargs=2, default=[1.0, 3.0], help='range of the smallest fault displacement at zoom 1, samples')
parser.add_argument('--disp_max', type=float, nargs=2, default=[6.0, 20.0], help='range of the largest fault displacement at zoom 1, samples')
parser.add_argument('--relief', type=float, nargs=2, default=[2.0, 30.0], help='range of the reflector relief at zoom 1, samples, log-uniform')
parser.add_argument('--noise', type=float, nargs=2, default=[0.0, 0.6], help='range of the noise level')
parser.add_argument('--fwidth', type=float, default=2.0, help='width of the fault labels, samples')
parser.add_argument('--p_none', type=float, default=0.05, help='probability of a model without faults')
parser.add_argument('--p_regular', type=float, default=0.1, help='probability of quasi-regular fault sets')
parser.add_argument('--p_thrust', type=float, default=0.1, help='probability of thrust faults')
parser.add_argument('--p_unconf', type=float, default=0.2, help='probability of unconformities')
parser.add_argument('--p_vary_disp', type=float, default=0.5, help='probability of faults with a displacement that varies along the fault')
parser.add_argument('--p_short', type=float, default=0.5,
                    help='share of the models with varying displacement whose slip patches are small, which gives short faults')
opts = parser.parse_args()


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


def generate(i, attempt=0):
    """Model i: image, labels, seismicity and a description of the model (None for a non-finite image)."""

    n1, n2, n3 = opts.n1, opts.n2, opts.n3
    rng = model_rng(i, attempt)

    s = loguniform(rng, opts.zoom)
    p = rgm.rgm3(n1=n1, n2=n2, n3=n3, seed=int(rng.integers(1, 2**30)))
    meta = {'zoom': s}

    # Wavelet and layers
    f0 = rng.uniform(opts.fs[0], opts.fs[1]) / s
    period = 1000.0 / f0
    p.f0 = f0
    p.wave = 'ricker'
    p.wave_phase = rng.uniform(-90.0, 90.0) if rng.random() < 0.2 else rng.uniform(-30.0, 30.0)
    nl = max(4, int(round(n1 / (period * loguniform(rng, opts.thick)))))
    p.nl = nl
    p.lwv = rng.uniform(0.0, 0.4)
    p.lwh = rng.uniform(0.0, 0.4)
    p.fwidth = opts.fwidth

    # Reflector shape
    relief = min(loguniform(rng, opts.relief) * s, 0.3 * n1)
    p.refl_slope = list(rng.uniform(-10.0, 10.0, 2))
    p.refl_height = [0.0, relief]
    p.refl_height_top = [0.0, min(rng.uniform(1.0, 5.0) * s, relief)]
    shape = ['gaussian', 'cauchy', 'perlin', 'random', 'fold', 'fold'][int(rng.integers(0, 6))]
    p.refl_shape = shape
    if shape in ('gaussian', 'cauchy'):
        p.ng = int(rng.integers(1, 4))
        p.rotate_fold = bool(rng.random() < 0.5)
        if rng.random() < 0.5:
            p.refl_skew = [0.3, 0.7]
    elif shape == 'perlin':
        p.refl_smooth = 4 * s
        p.refl_smooth_top = 10 * s
    elif shape == 'random':
        p.refl_smooth = 10 * s
        p.refl_smooth_top = 20 * s
    meta.update(f0=f0, nl=nl, shape=shape, relief=relief)

    # Faults: none, quasi-regular sets, thrusts with a few faults, or random faults
    r = rng.random()
    if r < opts.p_none:
        family = 'none'
    elif r < opts.p_none + opts.p_regular:
        family = 'regular'
    elif r < opts.p_none + opts.p_regular + opts.p_thrust:
        family = 'thrust'
    else:
        family = 'normal'

    disp_min = rng.uniform(opts.disp_min[0], opts.disp_min[1]) * s
    disp_max = max(disp_min, rng.uniform(opts.disp_max[0], opts.disp_max[1]) * s)
    p.delta_dip = [0.0, rng.uniform(0.0, 20.0)]
    p.delta_strike = [0.0, rng.uniform(0.0, 25.0)]
    p.rake = [60.0, 120.0]
    vary_disp = bool(rng.random() < opts.p_vary_disp)
    short = vary_disp and bool(rng.random() < opts.p_short)
    p.yn_vary_disp = vary_disp
    if short:
        p.disp_radius_dip = [0.15, 0.5]
        p.disp_radius_strike = [0.15, 0.5]
    p.yn_disp_decay = bool(rng.random() < 0.25)

    if family == 'none':
        nf = 0
    elif family == 'regular':
        p.yn_regular_fault = True
        nf = max(2, int(round(rng.integers(6, 15) / s)))
        st = rng.uniform(0.0, 180.0)
        p.strike = [st, st]
        if rng.random() < 0.5:
            p.dip = [rng.uniform(100.0, 120.0), rng.uniform(60.0, 80.0)]
        else:
            p.dip = [rng.uniform(60.0, 80.0), rng.uniform(100.0, 120.0)]
        p.disp = list(np.where(rng.random(2) < 0.5, 1.0, -1.0) * rng.uniform(2.0, 8.0) * s)
        p.rake = [90.0, 90.0]
    else:
        nf = max(1, int(round(rng.integers(opts.nf[0], opts.nf[1] + 1 + (4 if short else 0)) / s)))
        d = [disp_min, disp_max]
        p.disp = d if rng.random() < 0.5 else [-d[1], -d[0]]
        p.dip = [50.0, 130.0]
        p.strike = [0.0, 180.0]
        if family == 'thrust':
            p.nthrust = int(rng.integers(1, 4))
            p.thrust_dip = [15.0, 40.0]
            p.thrust_z = [1.0, 1.2]
            p.thrust_tip_z = [0.0, 0.5]
            nf = int(round(rng.integers(0, 5) / s))
    p.nf = nf
    meta.update(family=family, nf=nf, disp=[float(disp_min), float(disp_max)], vary_disp=vary_disp, short=short)

    if rng.random() < opts.p_unconf:
        unconf = int(rng.integers(1, 3))
        p.unconf = unconf
        p.unconf_z = [0.05, 0.35]
        p.unconf_height = [3.0 * s, 10.0 * s]
        meta['unconf'] = unconf

    # Imaging
    lat = rng.uniform(1.0, 3.0) * s
    p.psf_sigma = [12.0 * s, lat, lat]
    p.yn_dip_independent = bool(rng.random() < 0.25)
    noise_level = rng.uniform(opts.noise[0], opts.noise[1])
    p.noise_level = noise_level
    if rng.random() < 0.4:
        noise = 'migration'
    else:
        noise = ['normal', 'uniform'][int(rng.integers(0, 2))]
        ns = rng.uniform(0.0, 2.5, size=3)
        ns[ns < 0.5] = 0.0
        p.noise_smooth = [float(v * s) for v in ns]
    p.noise_type = noise
    if rng.random() < 0.5:
        p.illum_level = rng.uniform(0.0, 0.35)
        p.illum_smooth = 30.0 * s
    meta.update(noise=noise, noise_level=noise_level)

    p.generate()
    img = np.array(p.image)
    try:
        fault = np.array(p.fault)
        fdip = np.array(p.fault_dip) / 180.0
        fstrike = np.array(p.fault_strike) / 180.0
    except AttributeError:
        fault = np.zeros((n1, n2, n3), dtype=np.float32)
        fdip = np.zeros((n1, n2, n3), dtype=np.float32)
        fstrike = np.zeros((n1, n2, n3), dtype=np.float32)
    p.free()

    if not np.all(np.isfinite(img)):
        return None
    img = img / max(float(np.sqrt(np.mean(img**2))), 1.0e-30)

    # Seismicity of a volume: tens to thousands of events
    sim = simulate_seismicity(fault, seismicity_rng(i, attempt), scale=s, density=(0.002, 0.05), chain_density=(0.004, 0.03),
                              max_events=5000, sparse_events=(10, 101))
    meta['seismicity'] = sim['info']

    return img, fault, fdip, fstrike, sim, meta


def generate_checked(i, attempts=10):
    """Model i, with new random models in place of models with a non-finite image."""

    for attempt in range(attempts):
        out = generate(i, attempt)
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

            img, fault, fdip, fstrike, sim, meta = generate_checked(i)

            j, split = (i, 'train') if i < opts.ntrain else (i - opts.ntrain, 'valid')
            d = '%s/data_%s/%d' % (opts.outdir, split, j)
            t = '%s/target_%s/%d' % (opts.outdir, split, j)
            write_array(img, d + '_img.bin')
            write_array(sim['meq'], d + '_meq.bin')
            write_array(fault > 0, t + '_fsem.bin')
            write_array(fdip * (fault > 0), t + '_fdip.bin')
            write_array(fstrike * (fault > 0), t + '_fstrike.bin')
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

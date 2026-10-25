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
# Seismicity location images for SCF-Net in 2D and 3D (numpy and scipy only, so that a data generator
# can use it without importing torch, which corrupts the RGM images when imported before RGM):
#
#   render_seismicity     seismicity location image of event coordinates (Eq. 13 of Gao and Chen, 2025)
#   simulate_seismicity   random seismicity of a fault model, with inactive faults, patchy and
#                         clustered activity along faults, location errors, seismicity confined to
#                         a region, sparse catalogs and background events; it also returns the part
#                         of the faults that the seismicity indicates (the label of the
#                         seismicity-only mode)
#

import numpy as np
from scipy import ndimage


def render_seismicity_nd(x, shape, a=0.3, radius=4, chunk=2000):
    """
    Seismicity location image of events at the (fractional) grid coordinates x (a list of one array per
    axis, the first one along depth): S(x) = max_i exp(-a |x - x_i|^2), with a = 1/(2 sigma^2) = 0.3
    (Eq. 13 of Gao and Chen, 2025). Events outside the grid are ignored. Returns a float32 array.
    """
    nd = len(shape)
    s = np.zeros(shape, dtype=np.float32)
    x = [np.asarray(c, dtype=np.float64).ravel() for c in x]
    keep = np.ones(x[0].size, dtype=bool)
    for c, n in zip(x, shape):
        keep &= (c > -0.5) & (c < n - 0.5)
    x = [c[keep] for c in x]
    k = np.arange(-radius, radius + 1)
    for b in range(0, x[0].size, chunk):
        xc = [c[b:b + chunk] for c in x]
        idx = []
        for ax in range(nd):
            sh = [1] * (nd + 1)
            sh[0] = -1
            sh[ax + 1] = k.size
            idx.append((np.rint(xc[ax]).astype(np.int64).reshape([-1] + [1] * nd) + k.reshape(sh[1:]).reshape([1] + sh[1:])))
        idx = np.broadcast_arrays(*idx)
        r2 = sum((idx[ax] - xc[ax].reshape([-1] + [1] * nd))**2 for ax in range(nd))
        v = np.exp(-a * r2).astype(np.float32)
        inside = np.ones(v.shape, dtype=bool)
        for ax in range(nd):
            inside &= (idx[ax] >= 0) & (idx[ax] < shape[ax])
        np.maximum.at(s, tuple(i[inside] for i in idx), v[inside])
    return s


def render_seismicity(x1, x2, shape, a=0.3, radius=4):
    """Seismicity location image of events at the 2D grid coordinates x1 (depth) and x2 (see render_seismicity_nd)."""
    return render_seismicity_nd([x1, x2], shape, a, radius)


def smooth_field(shape, sigma, rng):
    """Zero-mean, unit-variance smooth random field."""
    g = ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode='wrap')
    return (g - g.mean()) / max(g.std(), 1.0e-12)


def loguniform(rng, a, b):
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


def simulate_seismicity(fault, rng, scale=1.0, scenario=None, p_scenario=(0.35, 0.25, 0.15, 0.25),
                        sigma_loc=(0.5, 8.0), density=(0.02, 0.5), max_sigma=12.0, min_pixels=8,
                        chain_density=(0.03, 0.15), chain_offset=(2.0, 8.0), chain_background=0.05, chain_active=(0.6, 1.0),
                        max_events=2000, sparse_events=(3, 31)):
    """
    Random seismicity of the faults in a fault index image or volume.

    fault:      (n1, n2) or (n1, n2, n3) fault index (0: no fault)
    scale:      grid refinement of the model (zoom); the location errors (but not the chain offsets) and the
                lengths of the activity patterns are scale times their values at scale 1
    scenario:   'full' (seismicity anywhere along the active faults), 'local' (only within a random
                elliptical region, e.g., around an injection well), 'sparse' (a few events), or 'chain'
                (chains of events within a few samples of most faults, with few background events, like
                the artificial seismicity of the published fine-tuning data and field examples, where
                every event lies within a few samples of a fault); drawn with probabilities p_scenario
                if None
    sigma_loc:  range of the location error at scale 1 (samples, log-uniform); field catalogs often scatter
                several samples around the faults of an image
    max_sigma:  largest lateral location error after the scaling (samples); depth errors are 1 to 2 times
                larger, and the errors of 5% of the events (outliers) 3 times larger
    max_events: largest number of fault events
    sparse_events: range of the number of events of a sparse catalog
    density:    range of the number of events per fault pixel that can host events (log-uniform)
    chain_*:    for the chain scenario, the ranges of the event density and of the largest offset of an
                event from its fault point (samples), the largest share of background events, and the
                range of the probability that a fault is active

    Every fault is active with a probability drawn from [0.3, 1]; inactive faults have no events.
    Along the active faults, a thresholded smooth random field gives patchy activity, and in half of
    the realizations a second one clusters the events; otherwise the events form chains along the
    faults. Events are located with Gaussian errors (larger in depth), with a few outliers, and
    background events that are not associated with faults are added.

    Returns a dictionary:
        meq       seismicity location image (Eq. 13)
        fsem_meq  fault pixels whose nearest event (true location) is on the same fault and within a distance
                  rlab = 6 + 1.5 sigma_loc samples, i.e., the part of the faults that the seismicity indicates
        fsem_act  all pixels of the active faults
        x1, x2    observed event locations (fault events and background events); x3 in 3D
        info      parameters of the realization
    """
    shape = fault.shape
    nd = len(shape)
    fault = np.rint(fault).astype(np.int64)
    ids, counts = np.unique(fault[fault > 0], return_counts=True)
    ids = ids[counts >= min_pixels]

    if scenario is None:
        scenario = ['full', 'local', 'sparse', 'chain'][int(rng.choice(4, p=np.asarray(p_scenario) / np.sum(p_scenario)))]
    chain = scenario == 'chain'

    info = {'scenario': scenario, 'scale': float(scale)}

    # Active faults
    p_act = rng.uniform(chain_active[0], chain_active[1]) if chain else rng.uniform(0.3, 1.0)
    active = ids[rng.random(ids.size) < p_act]
    if active.size == 0 and ids.size > 0:
        active = rng.choice(ids, 1)
    info['nfault'] = int(ids.size)
    info['nactive'] = int(active.size)
    act = np.isin(fault, active)

    # Region of the seismicity: an ellipse (2D) or ellipsoid (3D) of random orientation
    region = np.ones(shape, dtype=bool)
    if scenario == 'local':
        if np.any(act):
            # centered near an active fault, so that the cloud lights up part of a fault
            k = rng.integers(0, int(np.sum(act)))
            c = np.argwhere(act)[k]
        else:
            c = [rng.uniform(0, n) for n in shape]
        ax = [rng.uniform(0.1, 0.35) * n for n in shape]
        g = np.meshgrid(*[np.arange(n) - c0 for n, c0 in zip(shape, c)], indexing='ij')
        if nd == 2:
            th = rng.uniform(0, np.pi)
            u = g[0] * np.cos(th) + g[1] * np.sin(th)
            v = -g[0] * np.sin(th) + g[1] * np.cos(th)
            region = (u / ax[0])**2 + (v / ax[1])**2 <= 1.0
            info['region'] = [float(c[0]), float(c[1]), float(ax[0]), float(ax[1]), float(th)]
        else:
            rot, _ = np.linalg.qr(rng.standard_normal((nd, nd)))
            r = 0.0
            for m in range(nd):
                r = r + (sum(rot[m, n] * g[n] for n in range(nd)) / ax[m])**2
            region = r <= 1.0
            info['region'] = [float(v) for v in c] + [float(v) for v in ax]

    # Patchy activity along the faults and clustering of the events
    patch = smooth_field(shape, rng.uniform(4.0, 16.0) * scale, rng)
    q = rng.uniform(0.0, 0.4) if chain else rng.uniform(0.0, 0.5)
    patch = patch >= np.quantile(patch, q)
    cluster = smooth_field(shape, rng.uniform(3.0, 10.0) * scale, rng)
    beta = rng.uniform(0.0, 2.5) if (rng.random() < 0.5 and not chain) else 0.0

    host = act & region & patch
    nhost = int(np.sum(host))
    if scenario == 'sparse':
        nev = int(rng.integers(sparse_events[0], sparse_events[1]))
    elif chain:
        nev = int(round(loguniform(rng, chain_density[0], chain_density[1]) * nhost))
    else:
        nev = int(round(loguniform(rng, density[0], density[1]) * nhost))
    nev = min(nev, max_events) if nhost > 0 else 0
    info['nevent'] = nev

    # True locations: host pixels drawn with clustered weights, with sub-pixel jitter
    if chain:
        # largest distance of an event from its fault point, as in the published fine-tuning data
        sl = rng.uniform(chain_offset[0], chain_offset[1])
        r1 = 1.0
    else:
        sl = min(loguniform(rng, sigma_loc[0], sigma_loc[1]) * scale, max_sigma)
        r1 = rng.uniform(1.0, 2.0)
    info['sigma_loc'] = [float(sl * r1), float(sl)]
    if nev > 0:
        idx = np.argwhere(host)
        w = np.exp(beta * cluster[host])
        k = rng.choice(idx.shape[0], nev, p=w / np.sum(w))
        t = [idx[k, m] + rng.uniform(-0.5, 0.5, nev) for m in range(nd)]
        tid = fault[tuple(idx[k].T)]

        # Observed locations: Gaussian errors, larger in depth, and 5% outliers with 3 times the error;
        # for chains, a uniform offset within a box of a random half width between 0 and sl
        if chain:
            d = rng.uniform(0.0, sl, nev)
            x = [t[m] + rng.uniform(-1.0, 1.0, nev) * d for m in range(nd)]
        else:
            e = np.where(rng.random(nev) < 0.05, 3.0, 1.0)
            x = [t[m] + rng.standard_normal(nev) * sl * (r1 if m == 0 else 1.0) * e for m in range(nd)]
    else:
        t = [np.zeros(0) for m in range(nd)]
        x = [np.zeros(0) for m in range(nd)]
        tid = np.zeros(0, dtype=np.int64)

    # Background events not associated with faults, mostly uniform and sometimes clustered, within the region
    if scenario == 'sparse':
        nbg = int(rng.integers(0, 4))
    else:
        nbg = int(round(rng.uniform(0.0, chain_background if chain else 0.2) * max(nev, 10)))
    info['nbackground'] = nbg
    if nbg > 0:
        idx = np.argwhere(region)
        if chain or rng.random() < 0.75:
            k = rng.choice(idx.shape[0], nbg)
            b = [idx[k, m] + rng.uniform(-0.5, 0.5, nbg) for m in range(nd)]
        else:
            nc = int(rng.integers(1, 4))
            k = rng.choice(idx.shape[0], nc)
            c = rng.integers(0, nc, nbg)
            sc = rng.uniform(3.0, 12.0) * scale
            b = [idx[k[c], m] + rng.standard_normal(nbg) * sc for m in range(nd)]
        x = [np.concatenate((x[m], b[m])) for m in range(nd)]

    meq = render_seismicity_nd(x, shape)

    # Faults indicated by the seismicity: pixels of a fault whose nearest event (true location) is on the same
    # fault and within rlab
    fsem_meq = np.zeros(shape, dtype=np.float32)
    if nev > 0:
        rlab = 6.0 + 1.5 * sl
        ev = np.zeros(shape, dtype=np.int64)
        ev[tuple(np.clip(np.rint(t[m]).astype(np.int64), 0, shape[m] - 1) for m in range(nd))] = tid
        dist, near = ndimage.distance_transform_edt(ev == 0, return_indices=True)
        fsem_meq = ((fault > 0) & (fault == ev[tuple(near)]) & (dist <= rlab)).astype(np.float32)
        info['rlab'] = float(rlab)

    out = {'meq': meq, 'fsem_meq': fsem_meq, 'fsem_act': act.astype(np.float32), 'info': info}
    for m in range(nd):
        out['x%d' % (m + 1)] = x[m]
    return out


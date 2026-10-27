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
#
# Fault surfaces from the outputs of SCF-Net (fault probability, dip, and strike in 3D): fault polylines in 2D and
# triangulated fault surfaces in 3D:
#
#   cells    fault cells (Hale, 2013): samples of the smoothed fault probability that are maxima along the fault normal,
#            at a sub-sample position, with a value and an orientation
#   skins    fault skins (Hale, 2013): every cell is linked to its nearest neighbors along dip and strike when they are
#            mutually nearest, with similar values and orientations, near each other's planes; pieces of a fault are
#            merged across gaps of up to --max_gap samples (the main control of the continuity of the faults); faults
#            are the connected cells that hold a confident cell (--upper) and enough cells (--min_cells); last, every
#            cell moves along the fault normal onto the plane that fits its neighbors within --smooth_surface samples,
#            which removes the staircase of the sampling grid
#   mesh     3D: every fault is meshed on a regular grid of 1 sample over the plane that fits it (on several planes
#            where it bends), with gaps of up to --max_gap / 2 samples closed and holes of up to --max_hole square
#            samples filled
#   vote     optional (--vote), before the cells: optimal surface voting (Wu and Fomel, 2018), where around seeds on the
#            fault probability, dynamic programming finds the optimal surface (a path in 2D) through every seed in a
#            window aligned with the fault, and every surface votes with its mean probability; continuous faults
#            collect many votes and isolated discontinuities few. It suits a noisy fault probability: on a sharp one,
#            it gives about the same faults in 2D, and drops the smaller faults in 3D, as the votes scale with the size
#            of a fault
#
# This is an independent implementation of the published methods. The original methods assume steep faults (they thin
# along a horizontal direction and link cells one depth sample apart); here the thinning follows the fault normal and
# the linking the fault plane, so that low-angle thrusts are handled as well. Orientations follow the conventions of
# RGM and SCF-Net, with x1 downward: in 2D, the fault normal is (cos d, sin d) and the downward fault direction is
# (sin d, -cos d), for dip d; in 3D, the normal is (-cos d, sin d cos s, -sin d sin s), the dip vector
# (sin d, cos d cos s, -cos d sin s), and the strike vector (0, sin s, cos s), for dip d and strike s, in degrees.
#
# Usage (the inference must write the dip and strike wherever the probability may form cells, so run it with a
# --fault_threshold at most --lower, e.g., 0):
#
#   python src/fault_surfaces.py --fsem=predict.image.fsem --fdip=predict.image.fdip --n1=256 --n2=1024 \
#       --output=predict.image --image=image.bin
#   python src/fault_surfaces.py --fsem=predict.image.fsem --fdip=predict.image.fdip --fstrike=predict.image.fstrike \
#       --n1=128 --n2=256 --n3=256 --output=predict.image --image=image.bin
#
# Outputs, with x1 as the fastest axis of the raw files:
#
#   <output>.faults         index of the fault at the nearest sample of every cell, 0 elsewhere (float32)
#   <output>.faults.vtk     fault polylines (2D) or triangulated surfaces (3D) as legacy VTK polydata, with point
#                           attributes fault (index), probability (SCF-Net), dip[, strike][, score (with --vote)], which
#                           ParaView or pyvista display and color; points (x2, -x1, 0) in 2D and (x2, x3, -x1) in 3D,
#                           in samples, so that depth points down (the strike is the azimuth when x2 points east and x3
#                           north)
#   <output>.faults.h5      with --h5: the faults in HDF5: nvertex, the coordinates x1, x2[, x3] of the vertices in
#                           samples, their attributes (fault, probability, dip[, strike][, score]), and ntriangle and
#                           triangles (3D, ntriangle x 3) or nsegment and segments (2D, nsegment x 2), as 0-based
#                           indices of the vertices
#   <output>.events.vtk     with --events or --meq: the events as balls (legacy VTK polydata), with the attributes in
#                           the further columns of --events (the first of which colors the balls) and probability, the
#                           fault probability at the event; without --events, the events are the maxima of the
#                           location image --meq (events closer than about two samples merge into one)
#   <output>.volume.vti     with --volume: the image, the seismicity location image, and the fault probability, dip[,
#                           strike][, score] of the inference (in degrees), as a VTK image in the same frame
#   <output>.faults.txt     cells of the faults with the same attributes, as a table
#   <output>.faults.json    every fault: number of cells, mean probability, mean dip[ and strike], extent
#   <output>.faults.vote    with --vote: voting score (float32)
#   <output>.faults.png     with --image: the faults on the image (2D) or with image slices (3D), colored by --color
#
# References:
#   Hale, D., 2013, Methods to compute fault images, extract fault surfaces, and estimate fault throws from 3D seismic
#     images: Geophysics, 78(2), O33-O43.
#   Wu, X., and S. Fomel, 2018, Automatic fault interpretation with optimal surface voting: Geophysics, 83(5), O67-O82.
#

import argparse
import json

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree


def frames(dip, strike=None):
    """
    Unit vectors of faults from the dip (and strike in 3D) in degrees: (normal, downward fault direction) in 2D;
    (normal, dip vector, strike vector) in 3D; arrays of shape dip.shape + (ndim,).
    """

    d = np.radians(dip)
    if strike is None:
        return np.stack([np.cos(d), np.sin(d)], axis=-1), np.stack([np.sin(d), -np.cos(d)], axis=-1)
    s = np.radians(strike)
    n = np.stack([-np.cos(d), np.sin(d) * np.cos(s), -np.sin(d) * np.sin(s)], axis=-1)
    u = np.stack([np.sin(d), np.cos(d) * np.cos(s), -np.cos(d) * np.sin(s)], axis=-1)
    v = np.stack([np.zeros_like(s), np.sin(s), np.cos(s)], axis=-1)
    return n, u, v


def angles(n):
    """Dip (and strike in 3D) in degrees of unit normals n (..., ndim) of either sign; the strike is None in 2D."""

    if n.shape[-1] == 2:
        return np.mod(np.degrees(np.arctan2(n[..., 1], n[..., 0])), 180.0), None
    m = np.where((np.arctan2(-n[..., 2], n[..., 1]) < 0)[..., None], -n, n)
    strike = np.mod(np.degrees(np.arctan2(-m[..., 2], m[..., 1])), 180.0)
    dip = np.degrees(np.arctan2(np.hypot(m[..., 1], m[..., 2]), -m[..., 0]))
    return dip, strike


def sample(f, x):
    """Linear interpolation of the array f at positions x (..., ndim) in samples, clamped to the array."""

    xs = np.moveaxis(x, -1, 0).reshape(f.ndim, -1)
    return ndimage.map_coordinates(f, xs, order=1, mode='nearest').reshape(x.shape[:-1])


def gaussian_smooth(x, sigma):
    """x smoothed by a Gaussian of standard deviation sigma (samples), with periodic boundaries (via FFT)."""

    k2 = sum(k**2 for k in np.meshgrid(*[np.fft.fftfreq(n) for n in x.shape[:-1]], np.fft.rfftfreq(x.shape[-1]), indexing='ij'))
    return np.fft.irfftn(np.fft.rfftn(x) * np.exp(-2.0 * (np.pi * sigma)**2 * k2), s=x.shape)


STEPS = {2: np.array([[1, 0], [0, 1], [1, 1], [1, -1]]),
         3: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, -1], [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, -1, 0],
                      [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]])}


def find_cells(f, dip, strike, lower):
    """
    Fault cells of the image f: samples where f is a maximum along the grid direction nearest to the fault normal (of
    the 4 directions of the 8 neighbors in 2D, or the 13 of the 26 neighbors in 3D, so that two samples on either side
    of a ridge cannot both be maxima), moved to the vertex of the parabola through the three values along that
    direction, whose value there is at least lower. Returns positions (ncell, ndim), values and the indices of their
    samples.
    """

    idx = np.argwhere(f >= lower)
    n = frames(dip[tuple(idx.T)], None if strike is None else strike[tuple(idx.T)])[0]
    steps = STEPS[f.ndim]
    step = steps[np.argmax(np.abs(n @ (steps / np.linalg.norm(steps, axis=1, keepdims=True)).T), axis=1)]
    hi = np.array(f.shape) - 1
    f0 = f[tuple(idx.T)]
    fp, fm = f[tuple(np.clip(idx + step, 0, hi).T)], f[tuple(np.clip(idx - step, 0, hi).T)]
    keep = (f0 > fp) & (f0 >= fm)
    f1, f2 = 0.5 * (fp - fm), fp + fm - 2.0 * f0
    s = np.clip(np.where(f2 < 0, -f1 / np.where(f2 < 0, f2, -1.0), 0.0), -0.5, 0.5)
    fr = np.minimum(f0 + f1 * s + 0.5 * f2 * s * s, f.max())
    keep &= fr >= lower
    return idx[keep] + s[keep, None] * step[keep], fr[keep], idx[keep]


def pick_seeds(x, value, minimum, spacing, shape):
    """Seeds for voting: cells of value at least minimum, from the highest, with no other seed within spacing samples."""

    occupied = np.zeros(shape, dtype=bool)
    seeds = []
    for i in np.argsort(-value):
        if value[i] < minimum:
            break
        sl = tuple(slice(max(c - spacing, 0), c + spacing + 1) for c in np.round(x[i]).astype(int))
        if not occupied[sl].any():
            seeds.append(i)
            occupied[sl] = True
    return np.array(seeds, dtype=int)


def accumulate(c):
    """Minimum accumulated cost along the second-last axis, with a lag change (last axis) of at most 1 per step."""

    d = np.empty_like(c)
    d[..., 0, :] = c[..., 0, :]
    for k in range(1, c.shape[-2]):
        a = d[..., k - 1, :]
        a = np.minimum(a, np.minimum(np.concatenate([a[..., :1], a[..., :-1]], -1), np.concatenate([a[..., 1:], a[..., -1:]], -1)))
        d[..., k, :] = c[..., k, :] + a
    return d


def smooth_cost(c, axis):
    """Cost of the best path through every sample along axis: forward plus backward accumulation, minus the cost."""

    ct = np.moveaxis(c, axis, -2)
    return np.moveaxis(accumulate(ct) + accumulate(ct[..., ::-1, :])[..., ::-1, :] - ct, -2, axis)


def optimal_path(c):
    """Lag indices (..., K) of the path of minimum cost through c (..., K, L), with a lag change of at most 1 per step."""

    K = c.shape[-2]
    d = np.empty_like(c)
    choice = np.zeros(c.shape, dtype=np.int8)
    d[..., 0, :] = c[..., 0, :]
    inf = np.full(c.shape[:-2] + (1,), np.inf)
    for k in range(1, K):
        a = d[..., k - 1, :]
        opts = np.stack([np.concatenate([inf, a[..., :-1]], -1), a, np.concatenate([a[..., 1:], inf], -1)], -1)
        j = np.argmin(opts, -1)
        d[..., k, :] = c[..., k, :] + np.take_along_axis(opts, j[..., None], -1)[..., 0]
        choice[..., k, :] = j - 1
    path = np.empty(c.shape[:-1], dtype=int)
    path[..., K - 1] = np.argmin(d[..., K - 1, :], -1)
    for k in range(K - 1, 0, -1):
        path[..., k - 1] = path[..., k] + np.take_along_axis(choice[..., k, :], path[..., k:k + 1], -1)[..., 0]
    return path


def vote(p, dip, strike, seeds_x, radius, strain, smooth, batch=32):
    """
    Optimal surface voting (Wu and Fomel, 2018). Around every seed, a window spans radius[1] samples along dip, radius[2]
    along strike (3D) and radius[0] along the normal, with lags of 1/b samples (b = ceil(1/strain)), so that the surface
    slope is at most strain; the cost is 1 minus the smoothed probability, with lags beyond the distance from the seed
    excluded. The cost is smoothed by accumulation along dip (and strike), the optimal lag is picked along dip for every
    strike position, and the lag surface is smoothed by a Gaussian of smooth samples. Every surface votes with its mean
    smoothed probability at its samples, and every sample takes the orientation of its best voter, from the slopes of the
    lag surface at the seed. Returns the voting score, 1 - (1 - f)**8 with f the votes scaled to [0, 1], and the dip
    (and strike) of the best voters, or of the network where no surface reached.
    """

    nd, shape = p.ndim, p.shape
    fs = ndimage.gaussian_filter(p.astype(np.float64), 1.0)
    fs = (fs - fs.min()) / max(fs.max() - fs.min(), 1.0e-12)
    cost = 1.0 - fs
    b = int(np.ceil(1.0 / strain))
    rn, rd, rs = radius[0], radius[1], (radius[2] if nd == 3 else 0)
    lags = np.arange(-rn * b, rn * b + 1) / b
    kk, jj = np.arange(-rd, rd + 1, dtype=float), np.arange(-rs, rs + 1, dtype=float)
    r = np.hypot(jj[:, None], kk[None, :])
    excluded = np.abs(lags)[None, None, :] > np.where(r > 2, np.minimum(np.round(r), rn), 0)[:, :, None]
    xi = np.round(seeds_x).astype(int)
    frame = frames(dip[tuple(xi.T)], None if strike is None else strike[tuple(xi.T)])
    n0, u0 = frame[0], frame[1]
    v0 = frame[2] if nd == 3 else np.zeros_like(u0)
    flat, weight, sdip, sstrike = [], [], [], []
    for i0 in range(0, len(seeds_x), batch):
        sl = slice(i0, i0 + batch)
        xs, n, u, v = seeds_x[sl], n0[sl], u0[sl], v0[sl]
        base = xs[:, None, None, :] + jj[None, :, None, None] * v[:, None, None, :] + kk[None, None, :, None] * u[:, None, None, :]
        c = sample(cost, base[:, :, :, None, :] + lags[None, None, None, :, None] * n[:, None, None, None, :])
        c[:, excluded] = 1.0
        c = smooth_cost(c, 2)
        if nd == 3:
            c = smooth_cost(c, 1)
        s = lags[optimal_path(c)]
        if smooth > 0:
            s = ndimage.gaussian_filter(s, (0, smooth if nd == 3 else 0, smooth), mode='nearest')
        pts = base + s[..., None] * n[:, None, None, :]
        inside = np.all((pts >= 0) & (pts <= np.array(shape) - 1), axis=-1)
        fa = np.sum(sample(fs, pts) * inside, axis=(1, 2)) / np.maximum(np.sum(inside, axis=(1, 2)), 1)
        jc, kc = rs, rd
        nr = n - 0.5 * (s[:, jc, kc + 1] - s[:, jc, kc - 1])[:, None] * u
        if nd == 3:
            nr -= 0.5 * (s[:, jc + 1, kc] - s[:, jc - 1, kc])[:, None] * v
        nr /= np.linalg.norm(nr, axis=-1, keepdims=True)
        dr, sr = angles(nr)
        ids = np.round(pts).astype(int)
        for k in range(len(xs)):
            fl = np.ravel_multi_index(tuple(ids[k][inside[k]].T), shape)
            flat.append(fl)
            weight.append(np.full(len(fl), fa[k]))
            sdip.append(np.full(len(fl), dr[k]))
            if nd == 3:
                sstrike.append(np.full(len(fl), sr[k]))
    flat, weight = np.concatenate(flat), np.concatenate(weight)
    fe = ndimage.gaussian_filter(np.bincount(flat, weights=weight, minlength=p.size).reshape(shape), 0.7)
    f = (fe - fe.min()) / max(fe.max() - fe.min(), 1.0e-12)
    score = 1.0 - (1.0 - f)**8
    # orientation of the best voter at every sample
    order = np.lexsort((weight, flat))
    last = np.r_[np.nonzero(np.diff(flat[order]))[0], len(order) - 1]
    best = order[last]
    vdip = dip.copy()
    vdip.flat[flat[best]] = np.concatenate(sdip)[best]
    vstrike = None
    if nd == 3:
        vstrike = strike.copy()
        vstrike.flat[flat[best]] = np.concatenate(sstrike)[best]
    return score, vdip, vstrike


def link_cells(x, value, dip, strike, lower, max_dvalue, max_ddip, max_dstrike, max_planar, radius=1.8):
    """
    Links of every cell to its best neighbor in every in-plane direction: +-t in 2D; +-u (dip) and +-v (strike) in 3D.
    The best neighbor in a direction is the nearest cell within radius ahead in that direction (its offset points
    mostly along the direction, within a cone around it), so that a chain of cells never skips one; the link holds
    when the two cells are each other's best neighbors and agree as in Hale (2013): both values at least lower, values
    within max_dvalue, dips within max_ddip and (for steep faults) strikes within max_dstrike degrees, and each cell
    within max_planar samples of the other's plane. Returns an array (ncell, 2 * (ndim - 1)) of neighbor indices, -1
    where none.
    """

    nc, nd = x.shape
    frame = frames(dip, strike)
    nrm, axes = frame[0], frame[1:]
    ndir = 2 * len(axes)
    pairs = cKDTree(x).query_pairs(radius, output_type='ndarray')
    a = np.concatenate([pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([pairs[:, 1], pairs[:, 0]])
    d = x[b] - x[a]
    proj = np.stack([np.sum(d * e[a], -1) for e in axes], -1)
    k = np.argmax(np.abs(proj), -1)
    pk = np.take_along_axis(proj, k[:, None], -1)[:, 0]
    direction = 2 * k + (pk < 0)
    dist = np.sum(d * d, -1)
    ok = (np.abs(pk) >= 0.5) & (np.sqrt(np.maximum(dist - pk * pk, 0.0)) <= 0.5 + 0.5 * np.abs(pk))
    a, b, direction, dist = a[ok], b[ok], direction[ok], dist[ok]
    # best candidate of every cell and direction (no fallback to the next one, as in Hale, 2013)
    key = a * ndir + direction
    order = np.lexsort((dist, key))
    first = order[np.r_[0, np.nonzero(np.diff(key[order]))[0] + 1]]
    best = np.full((nc, ndir), -1, dtype=int)
    best[a[first], direction[first]] = b[first]
    # agreement of the two cells
    links = np.full((nc, ndir), -1, dtype=int)
    ca, cb = np.nonzero(best >= 0)
    nb = best[ca, cb]
    ax = cb // 2
    sgn = np.sign(np.sum(np.stack(axes)[ax, ca] * np.stack(axes)[ax, nb], -1))
    opposite = 2 * ax + np.where(sgn >= 0, 1 - cb % 2, cb % 2)
    good = best[nb, opposite] == ca
    good &= np.minimum(value[ca], value[nb]) >= lower
    good &= np.abs(value[ca] - value[nb]) <= max_dvalue
    good &= np.abs(dip_angle(nrm[ca]) - dip_angle(nrm[nb])) <= max_ddip
    if nd == 3:
        good &= strike_difference(nrm[ca], nrm[nb]) <= max_dstrike
    dd = x[nb] - x[ca]
    good &= np.maximum(np.abs(np.sum(nrm[ca] * dd, -1)), np.abs(np.sum(nrm[nb] * dd, -1))) <= max_planar
    links[ca[good], cb[good]] = nb[good]
    return links


def dip_angle(n):
    """Dip of the fault plane from horizontal (0 to 90 degrees) of unit normals n, or the direction (0 to 180) in 2D."""

    if n.shape[-1] == 2:
        return np.mod(np.degrees(np.arctan2(n[..., 1], n[..., 0])), 180.0)
    return np.degrees(np.arccos(np.minimum(np.abs(n[..., 0]), 1.0)))


def strike_difference(na, nb):
    """Difference of strike in degrees between the faults of unit normals na and nb (0 where either is nearly flat)."""

    sgn = np.sign(np.sum(na * nb, -1))[..., None]
    ha, hb = na[..., 1:], sgn * nb[..., 1:]
    norm = np.linalg.norm(ha, axis=-1) * np.linalg.norm(hb, axis=-1)
    c = np.sum(ha * hb, -1) / np.maximum(norm, 1.0e-12)
    return np.where(norm > 0.04, np.degrees(np.arccos(np.clip(c, -1.0, 1.0))), 0.0)


def smooth_normals(links, dip, strike, npass):
    """Normals of the cells, averaged npass times with those of their linked neighbors (half weight each pass)."""

    n = frames(dip, strike)[0]
    for _ in range(npass):
        m = np.zeros_like(n)
        cnt = np.zeros(len(n))
        for k in range(links.shape[1]):
            j = links[:, k]
            ok = j >= 0
            nb = n[j[ok]]
            m[ok] += np.sign(np.sum(n[ok] * nb, -1))[:, None] * nb
            cnt[ok] += 1
        has = cnt > 0
        n[has] = 0.5 * n[has] + 0.5 * m[has] / cnt[has, None]
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return angles(n)


def components(links):
    """Connected components of the linked cells (an index from 0 for every cell)."""

    a, k = np.nonzero(links >= 0)
    nc = len(links)
    graph = sparse.coo_matrix((np.ones(len(a)), (a, links[a, k])), shape=(nc, nc))
    return csgraph.connected_components(graph, directed=False)[1]


def bridge_gaps(x, dip, strike, links, comp, max_gap, max_ddip, max_dstrike, max_planar):
    """
    Merges pieces of faults across gaps: a cell without a neighbor in an in-plane direction is joined to the nearest
    cell of another piece ahead in that direction within max_gap samples (its offset points mostly along the direction,
    within a cone), when each cell lies within max_planar samples of the other's plane and their dips (and strikes)
    agree; two pieces merge into one when one such bridge joins them in 2D, or three in 3D (so that two faults that
    touch at a corner stay apart), and the two cells are linked when the second also has no neighbor toward the first,
    so that a fault line continues through the gap. Returns the links and the component of every cell.
    """

    if max_gap <= 0:
        return links, comp
    frame = frames(dip, strike)
    nrm, axes = frame[0], frame[1:]
    ndir = 2 * len(axes)
    ends = np.nonzero(np.any(links < 0, axis=1))[0]
    near = cKDTree(x).query_ball_point(x[ends], max_gap)
    c = np.repeat(ends, [len(l) for l in near])
    j = np.concatenate([np.asarray(l, dtype=int) for l in near]) if len(near) else np.zeros(0, dtype=int)
    other = comp[j] != comp[c]
    c, j = c[other], j[other]
    d = x[j] - x[c]
    proj = np.stack([np.sum(d * e[c], -1) for e in axes], -1)
    k = np.argmax(np.abs(proj), -1)
    pk = np.take_along_axis(proj, k[:, None], -1)[:, 0]
    direction = 2 * k + (pk < 0)
    lateral = np.sqrt(np.maximum(np.sum(proj**2, -1) - pk**2, 0.0))
    ok = (links[c, direction] < 0) & (np.abs(pk) >= 0.5) & (lateral <= 0.5 + 0.5 * np.abs(pk))
    ok &= np.maximum(np.abs(np.sum(nrm[c] * d, -1)), np.abs(np.sum(nrm[j] * d, -1))) <= max_planar
    ok &= np.abs(dip_angle(nrm[c]) - dip_angle(nrm[j])) <= max_ddip
    if len(axes) == 2:
        ok &= strike_difference(nrm[c], nrm[j]) <= max_dstrike
    c, j, direction, dist = c[ok], j[ok], direction[ok], np.sum(d[ok]**2, -1)
    if len(c) == 0:
        return links, comp
    key = c * ndir + direction
    order = np.lexsort((dist, key))
    first = order[np.r_[0, np.nonzero(np.diff(key[order]))[0] + 1]]
    c, j, direction, dist = c[first], j[first], direction[first], dist[first]
    ncomp = comp.max() + 1
    pair = np.minimum(comp[c], comp[j]) * ncomp + np.maximum(comp[c], comp[j])
    _, inverse, count = np.unique(pair, return_inverse=True, return_counts=True)
    supported = count[inverse] >= (1 if len(axes) == 1 else 3)
    c, j, direction, dist = c[supported], j[supported], direction[supported], dist[supported]
    graph = sparse.coo_matrix((np.ones(len(c)), (comp[c], comp[j])), shape=(ncomp, ncomp))
    comp = csgraph.connected_components(graph, directed=False)[1][comp]
    dj = x[c] - x[j]
    projb = np.stack([np.sum(dj * e[j], -1) for e in axes], -1)
    kb = np.argmax(np.abs(projb), -1)
    dirb = 2 * kb + (np.take_along_axis(projb, kb[:, None], -1)[:, 0] < 0)
    for i in np.argsort(dist):
        if links[c[i], direction[i]] < 0 and links[j[i], dirb[i]] < 0:
            links[c[i], direction[i]] = j[i]
            links[j[i], dirb[i]] = c[i]
    return links, comp


def local_planes(x, fault, radius):
    """
    Weighted mean and unit normal of the plane (line in 2D) that best fits the cells of the same fault within radius
    samples of every cell, with Gaussian weights of standard deviation radius / 2, and the number of these cells.
    """

    nk, nd = len(x), x.shape[1]
    pairs = cKDTree(x).query_pairs(radius, output_type='ndarray')
    pairs = pairs[fault[pairs[:, 0]] == fault[pairs[:, 1]]]
    a = np.concatenate([np.arange(nk), pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([np.arange(nk), pairs[:, 1], pairs[:, 0]])
    w = np.exp(-2.0 * np.sum((x[a] - x[b])**2, -1) / radius**2)
    s0 = np.bincount(a, weights=w, minlength=nk)
    m = np.stack([np.bincount(a, weights=w * x[b, i], minlength=nk) for i in range(nd)], -1) / s0[:, None]
    q = x[b] - m[a]
    cov = np.zeros((nk, nd, nd))
    for i in range(nd):
        for j in range(i, nd):
            cov[:, i, j] = cov[:, j, i] = np.bincount(a, weights=w * q[:, i] * q[:, j], minlength=nk)
    return m, np.linalg.eigh(cov)[1][:, :, 0], np.bincount(a, minlength=nk)


def smooth_surfaces(x, fault, radius, npass=3):
    """
    Cell positions smoothed over every fault: npass times, every cell of a fault moves along the local normal onto the
    plane (line in 2D) that best fits the cells of the same fault within radius samples (local_planes). This removes
    the staircase that the sampling grid leaves in the cells, and keeps the extent and the large-scale curvature of a
    fault.
    """

    x = x.copy()
    keep = np.nonzero(fault > 0)[0]
    if radius <= 0 or len(keep) == 0:
        return x
    for _ in range(npass):
        xk = x[keep]
        m, n, count = local_planes(xk, fault[keep], radius)
        move = count > x.shape[1]
        x[keep[move]] = xk[move] - np.sum((xk[move] - m[move]) * n[move], -1)[:, None] * n[move]
    return x


def select_faults(comp, value, upper, min_cells):
    """
    Fault skins: the components that hold a cell of value at least upper and at least min_cells cells, numbered from 1
    by decreasing size; 0 for other cells.
    """

    ncomp = comp.max() + 1
    size = np.bincount(comp, minlength=ncomp)
    seeded = np.bincount(comp, weights=(value >= upper).astype(float), minlength=ncomp) > 0
    kept = np.nonzero(seeded & (size >= min_cells))[0]
    kept = kept[np.argsort(-size[kept], kind='stable')]
    index = np.zeros(ncomp, dtype=int)
    index[kept] = np.arange(1, len(kept) + 1)
    return index[comp]


def chains(links, fault):
    """
    Ordered cells of the 2D faults: walks along the links from the ends of the chains (then from any cell, for closed
    chains) until every cell of every fault is visited; a fault merged across a gap without a link has several chains.
    Returns a list of (fault index, path of cells).
    """

    out = []
    visited = np.zeros(len(fault), dtype=bool)
    for start in np.lexsort((np.sum(links >= 0, axis=1), fault)):
        if fault[start] == 0 or visited[start]:
            continue
        path, prev, cur = [start], -1, start
        visited[start] = True
        while True:
            nxt = [j for j in links[cur] if j >= 0 and j != prev and not visited[j]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            visited[cur] = True
            path.append(cur)
        if len(path) > 1:
            out.append((fault[start], np.array(path)))
    return out


def mesh_faults(x, fault, normal, onormal, fields, max_hole, close=1, max_angle=60.0, min_patch=50, npass=3, sigma=1.0):
    """
    Triangulated surfaces of the 3D faults on regular grids. The cells of a fault are projected on the plane that fits
    them best, on pixels of 1 sample, where every cell is spread over the four pixels around it with bilinear weights:
    the surface lies at the mean height of the cells above the plane, smoothed by a Gaussian of sigma pixels; gaps of up to 2 close pixels are closed and holes of at most
    max_hole pixels filled, by interpolation; and every 2 x 2 block of pixels gives two triangles (one when a pixel is
    missing). Groups of at least min_patch cells whose local normal (normal) is more than max_angle degrees from the
    normal of the plane, where a curved fault would fold over it, are meshed again on the plane that fits them, with
    the cells within 2 samples, up to npass times. The scalars of the cells (fields)
    and their fault normals (onormal) are averaged in the same way. Returns the points (npoint, 3) in samples, the
    triangles (ntri, 3), and, at every point, the fields, the fault index and the unit fault normal.
    """

    cmin = np.cos(np.radians(max_angle))
    points, tris, nrm, fid_out = [], [], [], []
    out = {k: [] for k in fields}
    npt = 0
    for fid in range(1, fault.max() + 1):
        c = np.nonzero(fault == fid)[0]
        tree = cKDTree(x[c])
        left = c
        for it in range(npass):
            if len(left) < 3:
                break
            sel = left if it == 0 else np.unique(np.concatenate([left, c[np.concatenate(tree.query_ball_point(x[left], 2.0)).astype(int)]]))
            m = x[sel].mean(axis=0)
            e = np.linalg.svd(x[sel] - m, full_matrices=False)[2]
            bad = np.abs(normal[sel] @ e[2]) < cmin
            if bad.any():
                # small groups of such cells stay: their normals are less reliable than their positions
                xb = x[sel[bad]]
                pairs = cKDTree(xb).query_pairs(1.8, output_type='ndarray')
                graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xb), len(xb)))
                group = csgraph.connected_components(graph, directed=False)[1]
                bad[np.nonzero(bad)[0][np.bincount(group)[group] < min_patch]] = False
            p = sel[~bad]
            new = np.isin(left, p)
            if len(p) < 3 or not new.any():
                break
            left = left[~new]
            # cells on a grid of pixels over the plane
            d = x[p] - m
            u, v, h = d @ e[0], d @ e[1], d @ e[2]
            u0, v0 = np.floor(u.min()), np.floor(v.min())
            iu, iv = np.floor(u - u0).astype(int), np.floor(v - v0).astype(int)
            fu, fv = u - u0 - iu, v - v0 - iv
            nu, nv = iu.max() + 2, iv.max() + 2
            # every cell spread over the four pixels around it, with bilinear weights
            flat = np.concatenate([(iu + i) * nv + iv + j for i in (0, 1) for j in (0, 1)])
            wts = np.concatenate([(fu if i else 1.0 - fu) * (fv if j else 1.0 - fv) for i in (0, 1) for j in (0, 1)])
            sums = lambda w: np.bincount(flat, weights=wts * np.tile(w, 4), minlength=nu * nv).reshape(nu, nv)
            cnt = sums(np.ones(len(p)))
            occ = cnt >= 0.3
            # gaps of a pixel closed, and holes (empty regions inside the fault) of at most max_hole pixels filled
            k = close + 1
            closed = occ | ndimage.binary_closing(np.pad(occ, k), structure=np.ones((2 * close + 1, 2 * close + 1)))[k:-k, k:-k]
            bg, nbg = ndimage.label(~closed)
            fill = np.bincount(bg.ravel(), minlength=nbg + 1) <= max_hole
            fill[0] = False
            fill[np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))] = False
            mask = closed | fill[bg]
            # Gaussian-weighted means of the cells: over sigma pixels in the fault, and 3 sigma in the holes
            w1, w3 = ndimage.gaussian_filter(cnt, sigma, mode='constant'), ndimage.gaussian_filter(cnt, 3.0 * sigma, mode='constant')
            mean = lambda w: np.where(occ, ndimage.gaussian_filter(sums(w), sigma, mode='constant') / np.maximum(w1, 1.0e-12),
                                      ndimage.gaussian_filter(sums(w), 3.0 * sigma, mode='constant') / np.maximum(w3, 1.0e-12))[mask]
            gu, gv = np.nonzero(mask)
            points.append(m + np.outer(u0 + gu, e[0]) + np.outer(v0 + gv, e[1]) + np.outer(mean(h), e[2]))
            sgn = np.where(onormal[p] @ e[2] < 0, -1.0, 1.0)
            nn = np.stack([mean(sgn * onormal[p][:, k]) for k in range(3)], -1)
            nrm.append(nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1.0e-12))
            for k in fields:
                out[k].append(mean(fields[k][p]))
            fid_out.append(np.full(len(gu), fid))
            # two triangles per 2 x 2 block of pixels, or one with three pixels
            idx = np.full((nu, nv), -1)
            idx[mask] = npt + np.arange(len(gu))
            a, b, cc, dd = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
            A, B, C, D = a >= 0, b >= 0, cc >= 0, dd >= 0
            tris += [np.stack([a, b, dd], -1)[A & B & D], np.stack([a, dd, cc], -1)[A & D & C],
                     np.stack([a, b, cc], -1)[A & B & C & ~D], np.stack([b, dd, cc], -1)[B & D & C & ~A]]
            npt += len(gu)
    if npt == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), {}
    attrs = {'fault': np.concatenate(fid_out)}
    attrs.update({k: np.concatenate(out[k]) for k in fields})
    attrs['normal'] = np.concatenate(nrm)
    return np.concatenate(points), np.concatenate(tris), attrs


def display_frame(x):
    """
    Display coordinates of positions x (..., ndim) in samples, for ParaView or pyvista, with depth downward: (x2, -x1, 0)
    in 2D; (x2, x3, -x1) in 3D, where the strike is the azimuth when x2 points east and x3 north.
    """

    if x.shape[-1] == 2:
        return np.stack([x[..., 1], -x[..., 0], np.zeros(x.shape[:-1])], -1)
    return np.stack([x[..., 1], x[..., 2], -x[..., 0]], -1)


def display_grid(arrays):
    """Arrays (name: array of shape (n1, n2[, n3])) as a pyvista ImageData in the display frame of display_frame."""

    import pyvista as pv
    shape = next(iter(arrays.values())).shape
    n1 = shape[0]
    if len(shape) == 2:
        grid = pv.ImageData(dimensions=(shape[1], n1, 1), origin=(0.0, -(n1 - 1.0), 0.0))
        order = lambda a: a[::-1]
    else:
        grid = pv.ImageData(dimensions=(shape[1], shape[2], n1), origin=(0.0, 0.0, -(n1 - 1.0)))
        order = lambda a: a[::-1].transpose(0, 2, 1)
    for name, a in arrays.items():
        grid.point_data[name] = np.ascontiguousarray(order(a), dtype=np.float32).ravel()
    return grid


def find_events(s, threshold=0.75):
    """
    Event locations (nevent, ndim) in a seismicity location image s = max_i exp(-0.3 |x - x_i|^2) (Eq. 13 of Gao and
    Chen, 2025): its local maxima of at least threshold, refined along every axis by a parabola through the logarithm
    of s, which is exact for an isolated event; events closer than about two samples merge into one.
    """

    idx = np.argwhere((s >= threshold) & (s == ndimage.maximum_filter(s, size=3, mode='constant')))
    x = idx.astype(np.float64)
    hi = np.array(s.shape) - 1
    l0 = np.log(np.maximum(s[tuple(idx.T)], 1.0e-12))
    for ax in range(s.ndim):
        e = np.zeros(s.ndim, dtype=int)
        e[ax] = 1
        lp = np.log(np.maximum(s[tuple(np.minimum(idx + e, hi).T)], 1.0e-12))
        lm = np.log(np.maximum(s[tuple(np.maximum(idx - e, 0).T)], 1.0e-12))
        den = lp + lm - 2.0 * l0
        x[:, ax] += np.clip(np.where(den < 0, -0.5 * (lp - lm) / np.where(den < 0, den, -1.0), 0.0), -0.5, 0.5)
    return x


def write_events(filename, x, radius, point_data):
    """
    Events at positions x (nevent, ndim) as balls of radius samples in the display frame, as legacy VTK polydata, with
    the attributes point_data (one value per event), the first of which colors the balls by default.
    """

    import pyvista as pv
    cloud = pv.PolyData(display_frame(x))
    for name, val in point_data.items():
        cloud.point_data[name] = val
    balls = cloud.glyph(geom=pv.Sphere(radius=radius, theta_resolution=12, phi_resolution=12), scale=False, orient=False)
    for name in [k for k in balls.point_data.keys() if k not in point_data]:
        del balls.point_data[name]
    balls.set_active_scalars(next(iter(point_data)))
    balls.save(filename)


def write_vtk(filename, x, cells, point_data):
    """
    Fault polylines (2D) or triangulated surfaces (3D) as legacy VTK polydata: points in the display frame
    (display_frame), polylines or triangles (cells: a list or an array of point indices), and point arrays (name: value
    per point), which ParaView or pyvista display and color by attribute.
    """

    used = np.unique(np.concatenate([np.ravel(c) for c in cells]))
    remap = np.full(len(x), -1)
    remap[used] = np.arange(len(used))
    pts = display_frame(x[used])
    with open(filename, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nSCF-Net faults\nASCII\nDATASET POLYDATA\nPOINTS %d float\n' % len(used))
        np.savetxt(f, pts, fmt='%.3f')
        if x.shape[1] == 3:
            tri = remap[np.asarray(cells)]
            f.write('POLYGONS %d %d\n' % (len(tri), 4 * len(tri)))
            np.savetxt(f, np.column_stack([np.full(len(tri), 3), tri]), fmt='%d')
        else:
            f.write('LINES %d %d\n' % (len(cells), sum(len(c) + 1 for c in cells)))
            for c in cells:
                f.write('%d %s\n' % (len(c), ' '.join(str(i) for i in remap[c])))
        f.write('POINT_DATA %d\n' % len(used))
        for name, val in point_data.items():
            integer = np.issubdtype(val.dtype, np.integer)
            f.write('SCALARS %s %s 1\nLOOKUP_TABLE default\n' % (name, 'int' if integer else 'float'))
            np.savetxt(f, val[used], fmt='%d' if integer else '%.4f')


def write_h5(filename, x, cells, point_data):
    """
    Faults in an HDF5 file: the number of vertices (nvertex), their coordinates x1, x2[, x3] in samples (float32), their
    attributes (point_data, one array per name), and the connectivity of the triangles (3D, ntriangle x 3) or of the
    segments of the polylines (2D, nsegment x 2), as 0-based indices of the vertices (int32).
    """

    import h5py
    used = np.unique(np.concatenate([np.ravel(c) for c in cells]))
    remap = np.full(len(x), -1)
    remap[used] = np.arange(len(used))
    if x.shape[1] == 3:
        name, conn = 'triangle', remap[np.asarray(cells)]
    else:
        name, conn = 'segment', np.concatenate([np.stack([remap[c[:-1]], remap[c[1:]]], -1) for c in cells])
    with h5py.File(filename, 'w') as f:
        # a fixed-length ASCII string, which every HDF5 reader reads (unlike h5py's default variable-length UTF-8)
        f.attrs['description'] = np.bytes_('SCF-Net faults: vertices (coordinates x1 (depth), x2[, x3] in samples) with '
                                           'attributes, and %ss as 0-based indices of the vertices' % name)
        f.attrs['index_base'] = 0
        f['nvertex'] = np.int64(len(used))
        for i in range(x.shape[1]):
            f['x%d' % (i + 1)] = x[used, i].astype(np.float32)
        for k, v in point_data.items():
            f[k] = v[used].astype(np.int32 if np.issubdtype(v.dtype, np.integer) else np.float32)
        f['n%s' % name] = np.int64(len(conn))
        f['%ss' % name] = conn.astype(np.int32)


COLORS = ['#ff0000', '#0000ff', '#00b000', '#ff00ff', '#00c0c0', '#ff8000', '#8000ff', '#c0c000', '#000000', '#804000']
LABELS = {'fault': 'Fault', 'probability': 'Fault Probability', 'score': 'Voting Score', 'dip': 'Dip (Degrees)',
          'strike': 'Strike (Degrees)'}


def color_scale(name, values):
    """Colormap and range of an attribute: distinct colors for the fault index, jet otherwise."""

    from matplotlib.colors import ListedColormap
    if name == 'fault':
        nf = max(int(values.max()), 1)
        return ListedColormap([COLORS[i % len(COLORS)] for i in range(nf)]), (0.5, nf + 0.5)
    lim = {'probability': (0.0, 1.0), 'score': (0.0, 1.0), 'dip': (0.0, 180.0), 'strike': (0.0, 180.0)}[name]
    return 'jet', lim


def figure_2d(filename, image, x, paths, attribute, values):
    """Fault polylines on the image, colored by an attribute."""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 11})
    n1, n2 = image.shape
    fig, ax = plt.subplots(figsize=(min(12.0, 4.0 * n2 / n1 + 1.5), 4.0))
    v = 2.5 * np.sqrt(np.mean(image**2))
    ax.imshow(image, cmap='gray', vmin=-v, vmax=v, extent=(-0.5, n2 - 0.5, n1 - 0.5, -0.5), aspect='auto')
    cmap, lim = color_scale(attribute, values)
    segs = [np.column_stack([x[p[:-1], 1], x[p[:-1], 0], x[p[1:], 1], x[p[1:], 0]]).reshape(-1, 2, 2) for _, p in paths]
    vals = [0.5 * (values[p[:-1]] + values[p[1:]]) for _, p in paths]
    if segs:
        lc = LineCollection(np.concatenate(segs), cmap=cmap, linewidths=1.8)
        lc.set_array(np.concatenate(vals))
        lc.set_clim(*lim)
        ax.add_collection(lc)
        if attribute != 'fault':
            fig.colorbar(lc, ax=ax, pad=0.01, fraction=0.03).set_label(LABELS[attribute])
    ax.set_xlim(-0.5, n2 - 0.5)
    ax.set_ylim(n1 - 0.5, -0.5)
    ax.set_xlabel('X2 (Samples)')
    ax.set_ylabel('X1 (Samples)')
    ax.set_title('%d Faults' % len({fid for fid, _ in paths}))
    plt.tight_layout(pad=0.2)
    plt.savefig(filename, dpi=200)
    plt.close(fig)


def figure_3d(filename, image, vtkfile, attribute):
    """Fault surfaces with three image slices (pyvista, off screen), colored by an attribute."""

    import pyvista as pv
    n1, n2, n3 = image.shape
    grid = display_grid({'image': image})
    v = 2.5 * np.sqrt(np.mean(image**2))
    pl = pv.Plotter(off_screen=True, window_size=(1400, 1100))
    pl.set_background('white')
    # bottom of the volume, and its faces at x2 = 0 and x3 = 0
    for voi in [(0, n2 - 1, 0, n3 - 1, 0, 0), (0, 0, 0, n3 - 1, 0, n1 - 1), (0, n2 - 1, 0, 0, 0, n1 - 1)]:
        pl.add_mesh(grid.extract_subset(voi), scalars='image', cmap='gray', clim=(-v, v), show_scalar_bar=False)
    mesh = pv.read(vtkfile)
    if mesh.n_points > 0:
        cmap, lim = color_scale(attribute, mesh.point_data[attribute])
        pl.add_mesh(mesh, scalars=attribute, cmap=cmap, clim=lim, smooth_shading=True,
                    show_scalar_bar=attribute != 'fault',
                    scalar_bar_args={'title': LABELS[attribute], 'color': 'black', 'font_family': 'arial', 'n_labels': 5,
                                     'fmt': '%.0f' if attribute in ('dip', 'strike') else '%.1f'})
    pl.camera_position = [(2.4 * n2, 2.4 * n3, 1.6 * n1), (0.5 * n2, 0.5 * n3, -0.5 * n1), (0.0, 0.0, 1.0)]
    pl.screenshot(filename)
    pl.close()


def main():
    parser = argparse.ArgumentParser(description='Fault surfaces (3D) or polylines (2D) from SCF-Net outputs')
    parser.add_argument('--fsem', type=str, required=True, help='fault probability (float32, x1 fastest)')
    parser.add_argument('--fdip', type=str, required=True, help='fault dip / 180 (run the inference with --fault_threshold <= --lower)')
    parser.add_argument('--fstrike', type=str, default=None, help='fault strike / 180 (3D)')
    parser.add_argument('--image', type=str, default=None, help='seismic image, for the figure')
    parser.add_argument('--n1', type=int, required=True, help='number of samples along x1')
    parser.add_argument('--n2', type=int, required=True, help='number of samples along x2')
    parser.add_argument('--n3', type=int, default=0, help='number of samples along x3 (3D)')
    parser.add_argument('--output', type=str, required=True, help='prefix of the outputs')
    parser.add_argument('--vote', action='store_true', help='optimal surface voting before the cells (for a noisy fault probability)')
    parser.add_argument('--seed_min', type=float, default=0.3, help='voting: smallest smoothed probability of a seed')
    parser.add_argument('--spacing', type=int, default=4, help='voting: least distance between seeds, samples')
    parser.add_argument('--radius', type=int, nargs='+', default=None,
                        help='voting: half-widths of the window along the normal, dip (and strike), samples; '
                        'default 5 20 in 2D, 5 15 15 in 3D')
    parser.add_argument('--strain', type=float, default=0.25, help='voting: largest slope of the surface in the window')
    parser.add_argument('--smooth', type=float, default=2.0, help='voting: Gaussian smoothing of the surface, samples')
    parser.add_argument('--lower', type=float, default=None,
                        help='least value of a cell (default 0.2, or 0.65 with --vote)')
    parser.add_argument('--upper', type=float, default=None,
                        help='least value of the most confident cell of a fault (default 0.5, or 0.8 with --vote)')
    parser.add_argument('--min_cells', type=int, default=None, help='least number of cells of a fault (default 20 in 2D, 200 in 3D)')
    parser.add_argument('--max_dvalue', type=float, default=0.2, help='largest difference of value between linked cells')
    parser.add_argument('--max_ddip', type=float, default=10.0, help='largest difference of dip between linked cells, degrees')
    parser.add_argument('--max_dstrike', type=float, default=30.0, help='largest difference of strike between linked cells, degrees')
    parser.add_argument('--max_planar', type=float, default=1.0, help='largest distance of a linked cell from the plane of the other, samples')
    parser.add_argument('--max_gap', type=float, default=5.0,
                        help='longest gap, in samples, across which pieces of a fault are merged (0: no merging); '
                        'the main control of the continuity of the faults')
    parser.add_argument('--smooth_normals', type=int, default=2, help='passes of averaging of cell normals over linked neighbors')
    parser.add_argument('--max_hole', type=int, default=50,
                        help='3D: largest hole inside a fault surface that is filled, in square samples')
    parser.add_argument('--smooth_surface', type=float, default=4.0,
                        help='radius, in samples, of the local plane fits that smooth the faults (0: no smoothing)')
    parser.add_argument('--meq', type=str, default=None,
                        help='seismicity location image (float32, x1 fastest), as given to the inference; without --events, '
                        'its maxima give the events, written as balls')
    parser.add_argument('--events', type=str, default=None,
                        help='event locations, written as balls: a text or CSV file with x1, x2[ and x3] grid coordinates '
                        '(from 0) in its first columns, and attributes (e.g., the magnitude) in further columns, the first '
                        'of which colors the balls; as given to the inference with --input_events')
    parser.add_argument('--event_radius', type=float, default=1.0, help='radius of the balls of the events, samples')
    parser.add_argument('--h5', action='store_true',
                        help='also write the faults in an HDF5 file: vertices, their attributes, and the triangles (3D) or '
                        'segments (2D) as 0-based indices of the vertices')
    parser.add_argument('--volume', action='store_true',
                        help='also write the fault probability (and the image) as a VTK image in the frame of the faults')
    parser.add_argument('--color', type=str, default='fault', choices=['fault', 'probability', 'score', 'dip', 'strike'],
                        help='attribute that colors the faults in the figure')
    opts = parser.parse_args()
    assert opts.color != 'strike' or opts.n3 > 0, 'the strike is only available in 3D'
    assert opts.color != 'score' or opts.vote, 'the voting score is only available with --vote'

    shape = (opts.n1, opts.n2, opts.n3) if opts.n3 > 0 else (opts.n1, opts.n2)
    nd = len(shape)
    rd = lambda f: np.fromfile(f, dtype=np.float32).reshape(shape[::-1]).T.astype(np.float64)
    p = np.clip(rd(opts.fsem), 0.0, 1.0)
    dip = rd(opts.fdip) * 180.0
    strike = rd(opts.fstrike) * 180.0 if nd == 3 else None
    lower = opts.lower if opts.lower is not None else (0.65 if opts.vote else 0.2)
    upper = opts.upper if opts.upper is not None else (0.8 if opts.vote else 0.5)
    min_cells = opts.min_cells if opts.min_cells is not None else (20 if nd == 2 else 200)
    radius = opts.radius if opts.radius is not None else ([5, 20] if nd == 2 else [5, 15, 15])

    ps = ndimage.gaussian_filter(p, 1.0)
    if not opts.vote:
        score, vdip, vstrike = ps, dip, strike
    else:
        sx, sv, _ = find_cells(ps, dip, strike, opts.seed_min)
        seeds = pick_seeds(sx, sv, opts.seed_min, opts.spacing, shape)
        print('voting: %d seeds' % len(seeds))
        score, vdip, vstrike = vote(p, dip, strike, sx[seeds], radius, opts.strain, opts.smooth)
    if opts.vote:
        score.astype(np.float32).T.tofile(opts.output + '.faults.vote')

    x, value, idx = find_cells(score, vdip, vstrike, lower)
    cdip = vdip[tuple(idx.T)]
    cstrike = vstrike[tuple(idx.T)] if nd == 3 else None
    nzero = np.mean(cdip == 0)
    if nzero > 0.01:
        print('warning: %.0f%% of the cells have no dip: run the inference with --fault_threshold <= %g' % (100 * nzero, lower))
    links = link_cells(x, value, cdip, cstrike, lower, opts.max_dvalue, opts.max_ddip, opts.max_dstrike, opts.max_planar)
    cdip, cstrike = smooth_normals(links, cdip, cstrike, opts.smooth_normals)
    links, comp = bridge_gaps(x, cdip, cstrike, links, components(links), opts.max_gap, opts.max_ddip, opts.max_dstrike,
                              opts.max_planar)
    fault = select_faults(comp, value, upper, min_cells)
    nf = int(fault.max())
    print('%d cells, %d linked, %d faults' % (len(x), int(np.sum(np.any(links >= 0, axis=1))), nf))
    x = smooth_surfaces(x, fault, opts.smooth_surface)

    # outputs: attributes of every cell
    attrs = {'fault': fault, 'probability': sample(p, x), 'dip': cdip}
    if opts.vote:
        attrs['score'] = value
    if nd == 3:
        attrs['strike'] = cstrike
    names = [k for k in attrs if k != 'fault']
    keep = fault > 0
    label = np.zeros(shape, dtype=np.float32)
    label[tuple(np.clip(np.round(x[keep]).astype(int), 0, np.array(shape) - 1).T)] = fault[keep]
    label.T.tofile(opts.output + '.faults')
    np.savetxt(opts.output + '.faults.txt', np.column_stack([fault[keep], x[keep]] + [attrs[k][keep] for k in names]),
               fmt=['%d'] + ['%.3f'] * nd + ['%.4f'] * len(names),
               header='fault ' + ' '.join('x%d' % (i + 1) for i in range(nd)) + ' ' + ' '.join(names))
    summary = []
    n = frames(cdip, cstrike)[0]
    for fid in range(1, nf + 1):
        c = np.nonzero(fault == fid)[0]
        m = np.sum(np.sign(np.sum(n[c] * n[c[0]], -1))[:, None] * n[c], 0)
        fdip, fstrike = angles(m / np.linalg.norm(m))
        item = {'fault': fid, 'cells': int(len(c)), 'probability': round(float(attrs['probability'][c].mean()), 3),
                'dip': round(float(fdip), 1)}
        if opts.vote:
            item['score'] = round(float(value[c].mean()), 3)
        if nd == 3:
            item['strike'] = round(float(fstrike), 1)
        item.update({'min': [round(float(t), 1) for t in x[c].min(0)], 'max': [round(float(t), 1) for t in x[c].max(0)]})
        summary.append(item)
    with open(opts.output + '.faults.json', 'w') as f:
        json.dump(summary, f, indent=1)
    if nd == 2:
        paths = chains(links, fault)
        cells = [path for _, path in paths]
        if len(cells) > 0:
            write_vtk(opts.output + '.faults.vtk', x, cells, attrs)
            if opts.h5:
                write_h5(opts.output + '.faults.h5', x, cells, attrs)
    else:
        keep = np.nonzero(fault > 0)[0]
        geo = np.zeros_like(x)
        if len(keep) > 0:
            geo[keep] = local_planes(x[keep], fault[keep], 2.0 * (opts.smooth_surface if opts.smooth_surface > 0 else 4.0))[1]
        fields = {k: attrs[k] for k in attrs if k in ('probability', 'score')}
        pts, cells, mattrs = mesh_faults(x, fault, geo, frames(cdip, cstrike)[0], fields, opts.max_hole,
                                         close=max(1, int(round(0.5 * opts.max_gap))))
        pts = np.clip(pts, 0.0, np.array(shape) - 1.0)
        print('%d triangles' % len(cells))
        if len(cells) > 0:
            mdip, mstrike = angles(mattrs.pop('normal'))
            mattrs.update(dip=mdip, strike=mstrike)
            write_vtk(opts.output + '.faults.vtk', pts, cells, {k: mattrs[k] for k in attrs})
            if opts.h5:
                write_h5(opts.output + '.faults.h5', pts, cells, {k: mattrs[k] for k in attrs})
    image = rd(opts.image) if opts.image else None
    # seismicity: the given event locations (and their location image, without --meq), or the maxima of --meq
    seismicity = rd(opts.meq) if opts.meq else None
    events = None
    eattrs = {}
    if opts.events:
        from seismicity import read_events, render_seismicity_nd
        events, eattrs = read_events(opts.events, nd)
        if seismicity is None:
            seismicity = render_seismicity_nd([events[:, i] for i in range(nd)], shape).astype(np.float64)
    elif seismicity is not None:
        events = find_events(seismicity)
    if events is not None and len(events) > 0:
        # the first attribute of the file (e.g., its fourth column in 3D) colors the balls; then the fault probability
        write_events(opts.output + '.events.vtk', events, opts.event_radius, {**eattrs, 'probability': sample(p, events)})
        print('%d events' % len(events))
    if opts.volume:
        arrays = {} if image is None else {'image': image}
        if seismicity is not None:
            arrays['seismicity'] = seismicity
        arrays.update(probability=p, dip=dip)
        if nd == 3:
            arrays['strike'] = strike
        if opts.vote:
            arrays['score'] = score
        display_grid(arrays).save(opts.output + '.volume.vti')
    if image is not None:
        if nd == 2:
            figure_2d(opts.output + '.faults.png', image, x, paths, opts.color, attrs[opts.color])
        elif len(cells) > 0:
            figure_3d(opts.output + '.faults.png', image, opts.output + '.faults.vtk', opts.color)


if __name__ == '__main__':
    main()

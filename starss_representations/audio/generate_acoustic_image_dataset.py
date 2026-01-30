#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create acoustic map using APGD for an audio file.

We generate a .hdf file for every .wav file, containing 1) the acoustic image, 2) the associated metadata, 3) the
spatial audio, 4) the video frames.

By default, we use logarithmic spacing for frequency bands ranging from 50Hz to 4500Hz, with a total of 16 bands. We
use a timescale of 100 ms to match the labelling resolution of the DCASE files.
"""

import math
import time
import json
from argparse import ArgumentParser
from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import cv2
import pandas as pd
import librosa
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg
import scipy.io.wavfile as wavfile
import pyunlocbox as opt
from pyunlocbox.functions import dummy
from tqdm import tqdm
from scipy.constants import speed_of_sound
from scipy.signal.windows import tukey
from scipy.interpolate import griddata
from skimage.util import view_as_blocks, view_as_windows
from h5py import File

from starss_representations import utils

EIGENMIKE_COORDS = {
    "1": [69, 0, 0.042],
    "2": [90, 32, 0.042],
    "3": [111, 0, 0.042],
    "4": [90, 328, 0.042],
    "5": [32, 0, 0.042],
    "6": [55, 45, 0.042],
    "7": [90, 69, 0.042],
    "8": [125, 45, 0.042],
    "9": [148, 0, 0.042],
    "10": [125, 315, 0.042],
    "11": [90, 291, 0.042],
    "12": [55, 315, 0.042],
    "13": [21, 91, 0.042],
    "14": [58, 90, 0.042],
    "15": [121, 90, 0.042],
    "16": [159, 89, 0.042],
    "17": [69, 180, 0.042],
    "18": [90, 212, 0.042],
    "19": [111, 180, 0.042],
    "20": [90, 148, 0.042],
    "21": [32, 180, 0.042],
    "22": [55, 225, 0.042],
    "23": [90, 249, 0.042],
    "24": [125, 225, 0.042],
    "25": [148, 180, 0.042],
    "26": [125, 135, 0.042],
    "27": [90, 111, 0.042],
    "28": [55, 135, 0.042],
    "29": [21, 269, 0.042],
    "30": [58, 270, 0.042],
    "31": [122, 270, 0.042],
    "32": [159, 271, 0.042],
}
DEFAULT_EIGEN_DIRECTORY = utils.get_project_root() / "data/eigen_dev"
DEFAULT_OUTPATH = utils.get_project_root() / "outputs/apgd_dev"

#  Values taken from LAM paper
FMIN, FMAX = 1500, 4500
NBANDS = 9
# NBANDS = 2
SCALE = "linear"
BANDWIDTH = 50.0
# why does t_sti need to be 0.01 for 100 ms resolution? Shouldn't it be 0.1?
TSTI = 10e-3
FRAME_CAP = None
# FRAME_CAP = 10
SH_ORDER = 10
CIRCLE_RADIUS_DEG = 20
POLYGON_MASK_THRESHOLD = 4e-5
RESOLUTION = 640, 320


class L2Loss(opt.functions.func):
    r"""
    L2 loss function
    """

    def __init__(self, s, a):
        m, n = a.shape
        if not ((s.shape[0] == s.shape[1]) and (s.shape[0] == m)):
            raise ValueError('Parameters[S, A] are inconsistent.')
        if not np.allclose(s, s.conj().T):
            raise ValueError('Parameter[S] must be Hermitian.')

        super().__init__()
        self._S = s.copy()
        self._A = a.copy()

    def _eval(self, x):
        """
        Function evaluation.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        m, n = self._A.shape
        q = x.shape[1]
        b = ((self._A.reshape(1, m, n) * x.reshape(n, 1, q).T) @
             self._A.conj().T) - self._S

        z = np.sum(b * b.conj()).real
        return z

    def _grad(self, x):
        """
        Function gradient.
        """
        was_1d = (x.ndim == 1)
        if was_1d:
            x = x.reshape(-1, 1)

        m, n = self._A.shape
        q = x.shape[1]
        b = ((self._A.reshape(1, m, n) * x.reshape(n, 1, q).T) @
             self._A.conj().T) - self._S

        z = 2 * np.sum(self._A.conj() * (b @ self._A), axis=1).real.T
        if was_1d:
            z = z.reshape(-1)
        return z


class ElasticNetLoss(opt.functions.func):
    """
    Elastic-net regularizer.
    """

    def __init__(self, lambda_, gamma):
        if lambda_ < 0:
            raise ValueError('Parameter[lambda_] must be positive.')
        if not (0 <= gamma <= 1):
            raise ValueError('Parameter[gamma] must be in (0, 1).')

        super().__init__()
        self._lambda = lambda_
        self._gamma = gamma

    def _eval(self, x):
        """
        Function evaluation.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        l1_term = self._gamma * np.sum(np.abs(x), axis=0)
        l2_term = (1 - self._gamma) * np.sum(x ** 2, axis=0)

        z = np.sum(self._lambda * (l1_term + l2_term))
        return z

    def _prox(self, x, alpha):
        """
        Function proximal operator.
        """
        c1 = self._lambda * alpha * self._gamma
        c2 = 2 * self._lambda * alpha * (1 - self._gamma) + 1

        z = np.clip((x - c1) / c2, a_min=0, a_max=None)
        return z


class GroundTruthAccel(opt.acceleration.accel):
    """
    Acceleration scheme used to evaluate Acoustic Camera ground-truth.
    """

    def __init__(self, d, l, momentum=True):
        super().__init__()

        if d < 2:
            raise ValueError('Parameter[d] is out of range.')

        self._d = d
        self._step = 1 / l
        self._sol_prev = 0
        self._momentum = momentum

    def _pre(self, functions, x0):
        """
        Pre-processing specific to the acceleration scheme.
        """
        pass

    def _update_step(self, solver, objective, niter):  # noqa
        """
        Update the step size for the next iteration.
        """
        return self._step

    def _update_sol(self, solver, objective, niter):  # noqa
        """
        Update the solution point for the next iteration.
        """
        if self._momentum:
            step = (niter - 1) / (niter + self._d)
            sol = solver.sol + step * (solver.sol - self._sol_prev)
        else:
            sol = solver.sol
        self._sol_prev = solver.sol
        return sol

    def _post(self):
        """
        Post-processing specific to the acceleration scheme.
        """
        pass


def _degrees_to_radians(coords_dict: dict[str, list]) -> dict[str, list]:
    """
    Take a dictionary with microphone array capsules and 3D polar coordinates to convert them from degrees to radians
    colatitude, azimuth, and radius (radius is left intact)
    """
    return {
        m: [math.radians(c[0]), math.radians(c[1]), c[2]]
        for m, c in coords_dict.items()
    }


def _polar_to_cartesian(coords_dict: dict[str, list], units: str = None):
    """
    Take a dictionary with microphone array capsules and polar coordinates and convert to cartesian
    """
    if (
            units is None
            or not isinstance(units, str)
            or units.lower() not in ["degrees", "radians"]
    ):
        raise ValueError("Units must be specified as one of 'degrees' or 'radians'")
    elif units.lower() == "degrees":
        coords_dict = _degrees_to_radians(coords_dict)
    return {
        m: [
            c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0]),
        ]
        for m, c in coords_dict.items()
    }


def _equirectangular_to_cartesian(r, lat, lon):
    """
    Convert equirectangular values in form radius, latitude, longitude to cartesian
    """
    r = np.array([r])

    # Must be non-negative
    if np.any(r < 0):
        raise ValueError("Parameter `r` must be non-negative.")

    return (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
        .to_cartesian()
        .xyz.to_value(u.dimensionless_unscaled)
    )


def _equirectangular_to_spherical(
        x: int,
        y: int,
        width: int,
        height: int,
) -> tuple[int, int]:
    """
    Convert equirectangular pixel coordinates back to spherical coordinates.

    Arguments:
        x: Pixel x-coordinate
        y: Pixel y-coordinate
        width: Width in pixels
        height: Height in pixels

    Returns:
        (azimuth_deg, elevation_deg)
    """
    azimuth_deg = 180.0 - (x / width) * 360.0
    elevation_deg = 90.0 - (y / height) * 180.0
    return azimuth_deg, elevation_deg


def _cartesian_to_spherical(x: int, y: int, z: int) -> tuple[int, int]:
    """
    Convert Cartesian (x, y, z) to spherical (azimuth, elevation) in degrees.
    """
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(z))
    return azimuth, elevation


def _spherical_to_equirectangular(
        azimuth_deg: int,
        elevation_deg: int,
        width: int,
        height: int,
) -> tuple[int, int]:
    """
    Convert spherical coordinates to equirectangular pixel coordinates.

    Arguments:
        azimuth_deg: Azimuth in degrees [-180, 180]
        elevation_deg: Elevation in degrees [-90, 90]
        width: Width in pixels
        height: Height in pixels

    Returns:
        (x, y) pixel coordinates
    """
    # normalise azimuth from [-180, 180] to [0, img_width]
    #  azimuth 0° should be at centre (x = img_width/2)
    #  azimuth -180° should be at left edge (x = 0)
    #  azimuth +180° should be at right edge (x = img_width)
    x = ((-azimuth_deg + 180) % 360) / 360.0 * width

    # normalise elevation from [-90, 90] to [0, img_height]
    #  elevation +90° (up) should be at top (y = 0)
    #  elevation -90° (down) should be at bottom (y = img_height)
    y = (90 - elevation_deg) / 180.0 * height

    return int(x), int(y)


def get_xyz():
    mic_coords = _polar_to_cartesian(EIGENMIKE_COORDS, units='degrees')
    xyz = [[coo_ for coo_ in mic_coords[ch]] for ch in mic_coords]
    return xyz


def fibonacci(
        n: int,
        direction: np.ndarray = None,
        fo_v: float = None,
) -> np.ndarray:
    """
    Generate points on a unit sphere using Fibonacci lattice sampling.

    The Fibonacci lattice provides a nearly uniform distribution of points on a sphere's surface, making it ideal for
    spherical sampling applications. Points can optionally be limited to a specific region defined by a direction
    vector and field of view.

    Arguments:
        n (Numeric): Refinement level that determines the number of points. The total number of points generated is
            `4 * (n + 1)^2`.
        direction (np.ndarray, optional): A 3D unit vector specifying the central direction for region-limited
            sampling. Must be provided together with `fo_v`. If None, generates points over the entire sphere.
        fo_v (Numeric, optional): Field of view in radians, defining the angular extent of the region around
            `direction`. Must be in the range (0, 2π) or equivalently (0, 360) degrees. Required if `direction` is
            specified.

    Returns:
        np.ndarray: Array of shape (3, m) containing the Cartesian coordinates (x, y, z) of points on the unit sphere,
            where m ≤ 4 * (n + 1)^2. When region-limited, m is reduced to include only points within the specified FOV.
    """

    def _pol2cart(r, col, lo):
        lat = (np.pi / 2) - col
        return _equirectangular_to_cartesian(r, lat, lo)

    # This is the type of tesselation that we are using
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)

        if fo_v is not None:
            if not (0 < np.rad2deg(fo_v) < 360):
                raise ValueError("Parameter `fo_v` must be in (0, 360) degrees.")
        else:
            raise ValueError(
                "Parameter `fo_v` must be specified if `direction` is provided."
            )

    if n < 0:
        raise ValueError("Parameter `n` must be non-negative.")

    n_px = 4 * (n + 1) ** 2
    n = np.arange(n_px)

    colat = np.arccos(1 - (2 * n + 1) / n_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    xyz = np.stack(_pol2cart(1, colat, lon), axis=0)

    if direction is not None:  # region-limited case.
        # TODO: highly inefficient to generate the grid this way!
        min_similarity = np.cos(fo_v / 2)
        mask = (direction @ xyz) >= min_similarity
        xyz = xyz[:, mask]

    # these are the cartesian coordinates of the tesselation
    #  need to turn this into azimuth + elevation
    #  need to do the inverse of this: cart2pol
    #  sphere will have fewer points at the poles than expected
    #  to fill these, we need to do another interpolation
    return xyz


def get_field(sh_order: int = 10) -> np.ndarray:
    """
    Generate a hemisphere of sampling points for spherical harmonic field visualization.

    Creates a Fibonacci lattice on a unit sphere and filters it to retain only points within the upper hemisphere
    (i.e., z ≥ 0), with additional border trimming to avoid edge artifacts in visualization or processing.

    Arguments:
        sh_order (Numeric): Spherical harmonic order that determines sampling density. Higher orders produce more
            points. Defaults to `config.AIMG_SH_ORDER`. The initial grid contains `4 * (sh_order + 1)^2` points before
            filtering.

    Returns:
        np.ndarray: Array of shape (3, n) containing Cartesian coordinates (x, y, z) of points on the upper hemisphere.
    """

    # generate lattice
    r = fibonacci(sh_order)
    r_mask = np.abs(r[2, :]) < np.sin(np.deg2rad(90))
    r = r[:, r_mask]  # Shrink visible view to avoid border effects.
    # this is cartesian coordinates: (3, n_px)
    return r


def steering_operator(
        xyz: np.ndarray,
        r: int,
        fmin: int = FMIN,
        fmax: int = FMAX,
        n_bands: int = NBANDS,
) -> np.ndarray:
    """
    Compute steering matrix.
    """
    freq = np.linspace(fmin, fmax, n_bands)
    wl = speed_of_sound / (freq.max() + 500)
    if wl <= 0:
        raise ValueError(f"Parameter `wl` must be positive (got {wl}).")

    scale = 2 * np.pi / wl
    return np.exp((-1j * scale * xyz.T) @ r)


def extract_visibilities(
        data_: np.ndarray,
        rate_: int,
        t: int,
        fc: int,
        bw: int,
        alpha: float,
) -> np.ndarray:
    """
    Transform time-series to visibility matrices.
    """
    n_stft_sample = int(rate_ * t)
    if n_stft_sample == 0:
        raise ValueError("Not enough samples per time frame.")

    n_sample = (data_.shape[0] // n_stft_sample) * n_stft_sample
    n_channel = data_.shape[1]
    stf_data = view_as_blocks(
        data_[:n_sample], (n_stft_sample, n_channel)
    ).squeeze(
        axis=1
    )  # (n_stf, N_stft_sample, n_channel)

    window = tukey(M=n_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
    stf_win_data = stf_data * window  # (n_stf, N_stft_sample, n_channel)
    n_stf = stf_win_data.shape[0]

    stft_data = np.fft.fft(stf_win_data, axis=1)  # (n_stf, N_stft_sample, n_channel)
    # Find frequency channels to average together.
    idx_start = int((fc - 0.5 * bw) * n_stft_sample / rate_)
    idx_end = int((fc + 0.5 * bw) * n_stft_sample / rate_)
    collapsed_spectrum = np.sum(stft_data[:, idx_start: idx_end + 1, :], axis=1)

    # Don't understand yet why conj() on first term?
    return collapsed_spectrum.reshape(n_stf, -1, 1).conj() * collapsed_spectrum.reshape(
        n_stf, 1, -1
    )


# noinspection PyArgumentList
def eigh_max(a: np.ndarray):
    r"""
    Evaluate :math:`\mu_{\max}(\bbB)` with

    :math:
    B = (\overline{\bbA} \circ \bbA)^{H} (\overline{\bbA} \circ \bbA)
    """
    if a.ndim != 2:
        raise ValueError('Parameter[A] has wrong dimensions.')

    def matvec(v: np.ndarray):
        v = v.reshape(-1)
        c = (a * v) @ a.conj().T
        d = c @ a
        return np.sum(a.conj() * d, axis=0).real

    m, n = a.shape
    b = splinalg.LinearOperator(shape=(n, n), matvec=matvec, dtype=np.float64)
    d_max = splinalg.eigsh(b, k=1, which='LM', return_eigenvectors=False)
    return d_max[0]


def _solve(functions, x0, solver=None, atol=None, dtol=None, rtol=1e-3, xtol=None, maxit=200, verbosity='LOW'):
    """
    Solve an optimization problem whose objective function is the sum of some
    convex functions.
    """
    if verbosity not in ['NONE', 'LOW', 'HIGH', 'ALL']:
        raise ValueError('Verbosity should be either NONE, LOW, HIGH or ALL.')

    # Add a second dummy convex function if only one function is provided.
    if len(functions) < 1:
        raise ValueError('At least 1 convex function should be provided.')
    elif len(functions) == 1:
        functions.append(dummy())
        if verbosity in ['LOW', 'HIGH', 'ALL']:
            print('INFO: Dummy objective function added.')

    if not solver:
        raise ValueError("Solver function must be provided!")

    # Set solver and functions verbosity.
    translation = {'ALL': 'HIGH', 'HIGH': 'HIGH', 'LOW': 'LOW', 'NONE': 'NONE'}
    solver.verbosity = translation[verbosity]
    translation = {'ALL': 'HIGH', 'HIGH': 'LOW', 'LOW': 'NONE', 'NONE': 'NONE'}
    functions_verbosity = []
    for f in functions:
        functions_verbosity.append(f.verbosity)
        f.verbosity = translation[verbosity]

    tstart = time.time()
    crit = None
    niter = 0
    objective = [[f.eval(x0) for f in functions]]
    rtol_only_zeros = True

    # Solver specific initialization.
    solver.pre(functions, x0)
    tape_buffer = np.zeros((1000, len(x0)))
    tape_buffer[0] = x0

    while not crit:
        niter += 1

        if xtol is not None:
            last_sol = np.array(solver.sol, copy=True)

        if verbosity in ['HIGH', 'ALL']:
            name = solver.__class__.__name__
            print('Iteration {} of {}:'.format(niter, name))

        # Solver iterative algorithm.
        solver.algo(objective, niter)
        tape_buffer[niter] = solver.sol

        objective.append([f.eval(solver.sol) for f in functions])
        current = np.sum(objective[-1])
        last = np.sum(objective[-2])

        # Verify stopping criteria.
        if atol is not None and current < atol:
            crit = 'ATOL'
        if dtol is not None and np.abs(current - last) < dtol:
            crit = 'DTOL'
        if rtol is not None:
            div = current  # Prevent division by 0.
            if div == 0:
                if verbosity in ['LOW', 'HIGH', 'ALL']:
                    print('WARNING: (rtol) objective function is equal to 0 !')
                if last != 0:
                    div = last
                else:
                    div = 1.0  # Result will be zero anyway.
            else:
                rtol_only_zeros = False
            relative = np.abs((current - last) / div)
            if relative < rtol and not rtol_only_zeros:
                crit = 'RTOL'
        if xtol is not None:
            err = np.linalg.norm(solver.sol - last_sol)  # noqa
            err /= np.sqrt(last_sol.size)
            if err < xtol:
                crit = 'XTOL'
        if maxit is not None and niter >= maxit:
            crit = 'MAXIT'

        if verbosity in ['HIGH', 'ALL']:
            print('    objective = {:.2e}'.format(current))

    # Restore verbosity for functions. In case they are called outside solve().
    for k, f in enumerate(functions):
        f.verbosity = functions_verbosity[k]

    if verbosity in ['LOW', 'HIGH', 'ALL']:
        print('Solution found after {} iterations:'.format(niter))
        print('    objective function f(sol) = {:e}'.format(current))  # noqa
        print('    stopping criterion: {}'.format(crit))

    # Returned dictionary.
    result = {'sol': solver.sol,
              'solver': solver.__class__.__name__,  # algo for consistency ?
              'crit': crit,
              'niter': niter,
              'time': time.time() - tstart,
              'objective': objective}
    try:
        # Update dictionary for primal-dual solvers
        result['dual_sol'] = solver.dual_sol
    except AttributeError:
        pass

    # Solver specific post-processing (e.g. delete references).
    solver.post()

    result['backtrace'] = tape_buffer[:(niter + 1)]
    return result


def solve(s, a, lambda_=None, gamma=0.5, l=None, d=50, x0=None, eps=1e-3,
          n_iter_max=200, verbosity='LOW', momentum=True):
    """
    APGD solution to the Acoustic Camera problem.
    """
    m, n = a.shape
    if not ((s.shape[0] == s.shape[1]) and (s.shape[0] == m)):
        raise ValueError('Parameters[S, A] are inconsistent.')
    if not np.allclose(s, s.conj().T):
        raise ValueError('Parameter[S] must be Hermitian.')

    if not (0 <= gamma <= 1):
        raise ValueError('Parameter[gamma] is must lie in [0, 1].')

    if l is None:
        l = 2 * eigh_max(a)
    elif l <= 0:
        raise ValueError('Parameter[L] must be positive.')

    if d < 2:
        raise ValueError(r'Parameter[d] must be \ge 2.')

    if x0 is None:
        x0 = np.zeros((n,), dtype=np.float64)
    elif np.any(x0 < 0):
        raise ValueError('Parameter[x0] must be non-negative.')

    if not (0 < eps < 1):
        raise ValueError('Parameter[eps] must lie in (0, 1).')

    if n_iter_max < 1:
        raise ValueError('Parameter[N_iter_max] must be positive.')

    if verbosity not in ('NONE', 'LOW', 'HIGH', 'ALL'):
        raise ValueError('Unknown verbosity specification.')

    if lambda_ is None:
        if gamma > 0:  # Procedure of Remark 3.4
            # When gamma == 0, we fall into the ridge-regularizer case, so no
            # need to do the following.
            func = [L2Loss(s, a), ElasticNetLoss(lambda_=0, gamma=gamma)]
            solver = opt.solvers.forward_backward(accel=GroundTruthAccel(d, l, momentum=False))
            i_opt = _solve(functions=func,
                           x0=np.zeros((n,)),
                           solver=solver,
                           rtol=eps,
                           maxit=1,
                           verbosity=verbosity)
            alpha = 1 / l
            lambda_ = np.max(i_opt['sol']) / (10 * alpha * gamma)
        else:
            lambda_ = 1  # Anything will do.
    elif lambda_ < 0:
        raise ValueError('Parameter[lambda_] must be non-negative.')

    func = [L2Loss(s, a), ElasticNetLoss(lambda_, gamma)]
    solver = opt.solvers.forward_backward(accel=GroundTruthAccel(d, l, momentum))
    i_opt = _solve(functions=func,
                   x0=x0.copy(),
                   solver=solver,
                   rtol=eps,
                   maxit=n_iter_max,
                   verbosity=verbosity)
    i_opt['gamma'] = gamma
    i_opt['lambda_'] = lambda_
    i_opt['L'] = l
    return i_opt


def form_visibility(data, rate, fc, bw, t_sti, t_stationarity):
    """
    Visibilities computed directly in the frequency domain.
    """
    s_sti = (extract_visibilities(data, rate, t_sti, fc, bw, alpha=1.0))
    n_sample, n_channel = data.shape
    n_sti_per_stationary_block = int(t_stationarity / t_sti)
    return (
        view_as_windows(
            s_sti,
            window_shape=(n_sti_per_stationary_block, n_channel, n_channel),
            step=(n_sti_per_stationary_block, n_channel, n_channel)
        )
        .squeeze(axis=(1, 2))
        .sum(axis=1)
    )


def get_visibility_matrix(
        audio_in: np.ndarray,
        fs: int,
        t_sti: float = TSTI,
        scale: str = SCALE,
        nbands: int = NBANDS,
        frame_cap: int = FRAME_CAP,
        bw: int = BANDWIDTH,
) -> np.ndarray:
    """
    Compute visibility matrix from audio data.
    """

    print("\n=== get_visibility_matrix START ===")
    print(f"Sampling rate: {fs}")
    print(f"t_sti: {t_sti}")
    print(f"Scale: {scale}, nbands: {nbands}")

    # --- Frequency band creation ---
    print("[1/6] Computing frequency bands...")
    # Use spacing between 50 and 4500 Hz as in LAM paper
    if scale == "linear":
        freq = np.linspace(FMIN, FMAX, nbands)
    elif scale == "log":
        freq = librosa.mel_frequencies(n_mels=nbands, fmin=FMIN, fmax=FMAX)
    else:
        raise Exception("Not a valid scale to generate covariance matrices (log, linear)")
    print(f" → Freq bands: {freq.shape}, Bandwidth: {bw}")

    # --- Field and steering ---
    print("[2/6] Generating spherical field...")
    r = get_field()
    print(f" → Field shape: {r.shape}")

    print("[3/6] Loading Eigenmike geometry...")
    xyz = get_xyz()
    dev_xyz = np.array(xyz).T
    print(f" → Device XYZ shape: {dev_xyz.shape}")

    print("[4/6] Building steering operator...")
    a = steering_operator(dev_xyz, r)
    n_px = a.shape[1]
    print(f" → Steering matrix A: {a.shape}, n_px={n_px}")

    apgd_map = []

    print("[5/6] Processing bands...")
    for i in range(nbands):
        print(f"\n--- Band {i + 1}/{nbands} | freq={freq[i]:.2f} Hz ---")

        t_stationarity = 10 * t_sti
        print("    → Computing visibility matrices...")
        s = form_visibility(audio_in, fs, freq[i], bw, t_sti, t_stationarity)
        n_sample = s.shape[0]

        # Cap frames if required
        if frame_cap:
            s = s[:frame_cap, :, :]
            n_sample = frame_cap

        print(f"    → Visibility frames: {n_sample}")

        apgd_gamma = 0.5
        apgd_per_band = np.zeros((n_sample, n_px))
        i_prev = np.zeros((n_px,))

        print("    → Processing frames...")
        for s_idx in range(n_sample):

            if s_idx % 20 == 0 or s_idx == n_sample - 1:
                print(f"        Frame {s_idx + 1}/{n_sample}")

            # Eigen-decomposition
            s_d, s_v = linalg.eigh(s[s_idx])
            if s_d.max() <= 0:
                s_d[:] = 0
            else:
                s_d = np.clip(s_d / s_d.max(), 0, None)
            s_norm = (s_v * s_d) @ s_v.conj().T

            i_apgd = solve(
                s_norm,
                a,
                gamma=apgd_gamma,
                x0=i_prev.copy(),
                verbosity='NONE'
            )
            apgd_per_band[s_idx] = i_apgd['sol']
            i_prev = i_apgd['sol']

        apgd_map.append(apgd_per_band)

    print("\n[6/6] Final assembly...")
    # bands, frames, number of interpolated pixels -> need the tesselation
    apgd_arr = np.array(apgd_map)

    print(f" → APGD map shape: {apgd_arr.shape}")
    print("=== get_visibility_matrix END ===\n")

    return apgd_arr


def create_target_grid(width: int, height: int) -> np.ndarray:
    """
    Create regular target grid of points based on given width and height
    """
    target_az = np.linspace(
        180, -180, width
    )
    target_el = np.linspace(
        90, -90, height
    )
    target_az_grid, target_el_grid = np.meshgrid(target_az, target_el, indexing="xy")
    return np.stack([target_az_grid.ravel(), target_el_grid.ravel()], axis=1)


def create_2d_gaussian(
        cx: int,
        cy: int,
        width: int,
        height: int,
        circle_radius: int = CIRCLE_RADIUS_DEG,
) -> np.ndarray:
    """
    Compute a 2D Gaussian centered at `cx, cy` pixels.

    The radius of the circle is set to contain 2 SD of the values within the span of (width, height)
    """
    # Check inputs are valid
    if not 0 <= cx <= width:
        raise ValueError(
            f"X coordinate is outside of width! (x = {cx}, width = {width})"
        )
    if not 0 <= cy <= height:
        raise ValueError(
            f"Y coordinate is outside of height! (y = {cy}, height = {height})"
        )

    # The circle should contain 2 SD of the vals (68-*95*-99.7% rule)
    sigma_deg = circle_radius / 2.0

    deg_per_pixel_x = 360.0 / width
    deg_per_pixel_y = 180.0 / height

    _, center_elevation_deg = _equirectangular_to_spherical(
        cx, cy, width=width, height=height
    )

    x, y = np.arange(width), np.arange(height)
    xx, yy = np.meshgrid(x, y, indexing="xy")  # (H, W)

    # Wrapped pixel deltas (preserve sign)
    dx = (xx - cx + width / 2) % width - width / 2
    dy = yy - cy

    # Convert to angular deltas
    delta_az_deg = -dx * deg_per_pixel_x  # azimuth increases leftward
    delta_el_deg = dy * deg_per_pixel_y

    cos_lat = np.cos(np.radians(center_elevation_deg))

    dist_sq_deg = (delta_el_deg ** 2) + (cos_lat * delta_az_deg) ** 2

    gaussian = np.exp(-dist_sq_deg / (2.0 * sigma_deg ** 2))

    return gaussian


def find_contours(acoustic_image: np.ndarray) -> list[np.ndarray]:
    """
    Find contours in an equirectangular image. Horizontal wrap-around is handled naturally:
      - If a blob is split across left/right edges, findContours returns two separate contours
      - Both contours are kept in the segmentation list

    Args:
        acoustic_image (np.ndarray): 2D acoustic image (already scaled/masked)

    Returns:
        list[np.ndarray]: list of contours
    """
    # Binary mask
    binary_mask = (acoustic_image > 0).astype(np.uint8) * 255

    # Find contours normally
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # [(n_coordinates, 2), (n_coordinates, 2), ...]
    #  this will be len == 1 in all cases BUT when a sound event wraps around both edges of an image
    return [c.squeeze() for c in contours]


def get_segmentation_pixels(
        acoustic_image: np.ndarray, contour_boundary: np.ndarray
) -> list[list]:
    """
    Given an acoustic image and contour, compute pixel coordinate values of contour and return list of lists with
    inner structure [x_coord, y_coord, amplitude]
    """
    # We can just grab height and width from the acoustic image directly
    height, width = acoustic_image.shape

    # Compute the mask and fill with the contour boundary
    mask__ = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask__, [contour_boundary.astype(np.int32)], 255)

    # Stack to get (x, y, amplitude), with shape (N_coordinates, 3)
    y_coords, x_coords = np.where(mask__ == 255)
    amplitude_values = acoustic_image[y_coords, x_coords]
    pixels_data = np.column_stack([x_coords, y_coords, amplitude_values])

    # Return as a list of [x_coord, y_coord, amplitude] lists
    return [[int(x), int(y), amp] for (x, y, amp) in pixels_data.tolist()]


def generate_acoustic_image_json(
        acoustic_image: np.ndarray,
        metadata: np.ndarray,
        resolution: tuple[int, int] = RESOLUTION,
        polygon_mask_threshold: float = POLYGON_MASK_THRESHOLD,
        circle_radius: int = CIRCLE_RADIUS_DEG,
) -> list[dict]:
    """
    Generates a list of dictionaries (JSON-style) for a given acoustic image.

    The function presupposes both an acoustic image with shape (tesselation, bands, frames) and an array of metadata
    (computed using `audiblelight.synthesize.generate_dcase2024_metadata`, or similar). The method used is as follows:
        1. Take the median energy for each band in the acoustic image: gives (tesselation, frames)
        2. Iterate over all frames with an active annotation in the metadata array
            2a. Interpolate the corresponding acoustic image frame to an image with shape (height, width)
            2b. Iterate over all annotations for the current frame:
                2bi. Create a 2D Gaussian centered at the X and Y pixel coordinates of the annotation, with radius set
                        to span 2SD of all pixel values
                2bii. Scale the acoustic image frame by multiplying by the Gaussian
                2biii. Mask all values in the scaled acoustic image frame that are below `polygon_mask_threshold`
                2biv. Apply contour detection to grab the edges of each "blob" in the image
            2c. Append all "blobs" for the frame: each of these have the format [x_pixel, y_pixel, amplitude]
        3. Return a full dictionary containing annotations of every frame

    The dictionaries contain the following keys:
        - "metadata_frame_index": the index of the frame within the acoustic image
        - "instance_id": a unique integer identifier for each event in the scene
        - "category_id": the index of the soundevent
        - "distance": the distance of the soundevent
        - "segmentation": a list of [x_pixel, y_pixel, amplitude] values for every segmentation in that frame.

    Note that all but the last value are taken directly from the metadata: only "segemntation" is defined by the
    acoustic image.

    Finally, it is also assumed that the amplitude values should be scaled *across* multiple JSON files that constitute
    an entire dataset, e.g. by Z-scoring, scaling between 0 and 1, etc. As this process relies on summary statistics
    that cannot easily be known when computing individual JSONs, this must be accomplished after calling this function.

    Arguments:
        acoustic_image (np.ndarray): an acoustic image with shape (tesselation, bands, frames)
        metadata (np.ndarray): an array of metadata corresponding to the acoustic image
        resolution (tuple): the resolution to interpolate the image to: must be equirectangular, in form (width, height)
        polygon_mask_threshold (Numeric): after scaling the acoustic image according to the 2D Gaussian, values below
            this threshold will be set to 0. This value should be tweaked based on looking at the shape of the images.
        circle_radius (Numeric): the radius of the circle placed at ground-truth azimuth and elevation points when
            calculating the 2D Gaussian

    Returns:
        list[dict]: the metadata extracted for this acoustic image
    """
    # Validate the acoustic image
    if not acoustic_image.ndim == 3:
        raise ValueError(
            f"Expected acoustic image to have 3 dimensions, but got {acoustic_image.shape}"
        )

    # Store scene-wide results here
    scene_res = []

    # Unpack acoustic image
    n_tesselation, n_bands, n_frames = acoustic_image.shape

    # Compute median over bands once for entire acoustic image: shape (tesselation, frames)
    acoustic_image_medianed = np.median(acoustic_image, axis=1)

    # We can infer the `sh_order` used directly from the acoustic image, there's no need to pass it as an argument
    sh_order = int(math.sqrt(n_tesselation) / 2 - 1)

    # Get the tesselation coordinates and convert to spherical: shape (n_px, 2)
    tesselation = fibonacci(sh_order).T
    tesselation_eq = np.apply_along_axis(
        lambda x: _cartesian_to_spherical(*x), 1, tesselation
    )

    # Unpack video resolution
    video_width, video_height = resolution

    # Create regular target grid based on (scaled) width and height
    target_points = create_target_grid(video_width, video_height)

    # Grab frames with ground truth annotations and iterate over these
    frames_with_gt_annotations = np.unique(metadata[:, 0])
    for metadata_frame_idx in frames_with_gt_annotations:

        # Grab the corresponding acoustic image frame
        if metadata_frame_idx >= acoustic_image_medianed.shape[-1]:
            break

        acoustic_image_frame = acoustic_image_medianed[:, metadata_frame_idx]

        # Interpolate the acoustic image for this frame and reshape to (height, width)
        acoustic_image_interpolated = griddata(
            tesselation_eq,
            acoustic_image_frame,
            target_points,
            method="linear",
            fill_value=0.0,
        ).reshape(video_height, video_width)

        # Grab the annotations for this frame and iterate over
        #  We can have multiple annotations per frame, so this will be an array with min len == 1
        current_frame_metadatas = metadata[metadata[:, 0] == metadata_frame_idx]
        for metadata_row in current_frame_metadatas:

            # Grab everything from the row of metadata
            _, class_id, instance_id, gt_az, gt_el, gt_dist = metadata_row[:6]

            # Convert spherical azimuth/elevation to equirectangular
            gt_az_eq, gt_el_eq = _spherical_to_equirectangular(
                gt_az, gt_el, width=video_width, height=video_height
            )

            # Compute the 2D Gaussian centered at (azimuth, elevation): shape (width, height)
            gauss_gt = create_2d_gaussian(
                gt_az_eq,
                gt_el_eq,
                width=video_width,
                height=video_height,
                circle_radius=circle_radius,
            )

            # Multiply the acoustic image by the Gaussian to scale it
            acoustic_image_gauss_scaled = acoustic_image_interpolated * gauss_gt

            # Mask values in the scaled image that are below the threshold
            acoustic_image_gauss_masked = acoustic_image_gauss_scaled.copy()
            polygon_mask = np.where(
                acoustic_image_gauss_masked < polygon_mask_threshold
            )
            acoustic_image_gauss_masked[polygon_mask] = 0

            # Find contours within the masked image
            contours = find_contours(acoustic_image_gauss_masked)

            # We'll store segmentations for this frame inside here
            segmentations = []

            # Iterate over all the contours we've found
            for contour in contours:

                # skip degenerate contours
                if contour.ndim == 1:
                    continue

                # Grab the pixels + amplitude values within this segmentation and append to the list
                pixels_list = get_segmentation_pixels(
                    acoustic_image_gauss_masked, contour
                )
                segmentations.append(pixels_list)

            # Now we can create the annotations dictionary
            annotations_dict = {
                "metadata_frame_index": int(metadata_frame_idx),
                "instance_id": int(instance_id),
                "category_id": int(class_id),
                "segmentation": segmentations,
                "distance": float(gt_dist),
            }
            scene_res.append(annotations_dict)

    return scene_res


def main(data_src: str, outpath: str) -> None:
    eigenmike_files = [p for p in Path(data_src).rglob("**/*.wav") if "._" not in str(p)]

    # Sanitise output directory
    outdir = Path(outpath)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # store pixel amplitude distribution here
    pixel_amps = []

    # Iterate over every audio file
    for hdf_idx, clip_name in tqdm(enumerate(eigenmike_files[:2]), total=len(eigenmike_files), desc="Processing files..."):

        # Load in the WAV file
        sr, eigen_sig = wavfile.read(clip_name)

        # Create folder for split if required
        file_split = clip_name.parent.stem
        outpath_with_split = outdir / file_split
        if not outpath_with_split.exists():
            outpath_with_split.mkdir(parents=True)

        # load up the video for this file
        video_path = str(clip_name.with_suffix(".mp4")).replace("eigen_dev", "video_dev", )
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file")

        # get attributes from the video
        video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))

        # load in metadata for this file
        metadata_path = str(clip_name.with_suffix(".csv")).replace("eigen_dev", "metadata_dev")
        metadata = pd.read_csv(metadata_path)
        metadata.columns = ['frame', 'class_idx', 'source_idx', 'azimuth', 'elevation', 'distance']
        metadata["unique_source"] = metadata.groupby(["class_idx", "source_idx"]).ngroup()
        metadata = metadata.to_numpy()

        # Set filepath for this clip
        file_outpath = outpath_with_split / clip_name.with_suffix(".hdf").name
        print(f"Dumping HDF file to {file_outpath}...")

        # compute visibility graph matrix (32ch)
        apgd = get_visibility_matrix(
            eigen_sig,
            sr,
            t_sti=TSTI,
            scale=SCALE,
            nbands=NBANDS,
            bw=BANDWIDTH
        )

        # (tesselation, bands, frames)
        a_np = apgd.transpose((2, 0, 1))

        with File(file_outpath, "w") as f:
            # overall metadata
            f.attrs["file"] = clip_name.stem

            # acoustic image
            f.create_dataset("ai_apgd", shape=a_np.shape, dtype=a_np.dtype, data=a_np)
            f.attrs["ai_n_frames"] = a_np.shape[0]
            f.attrs["ai_n_bands"] = a_np.shape[1]

            # audio
            f.create_dataset("audio", shape=eigen_sig.shape, dtype=eigen_sig.dtype, data=eigen_sig)
            f.attrs["audio_sr"] = sr
            f.attrs["audio_duration"] = len(eigen_sig) / sr
            f.attrs["audio_n_frames"] = f.attrs["audio_duration"] / (TSTI * 10)
            f.attrs["audio_fpath"] = str(clip_name)

            # metadata
            f.create_dataset("metadata", shape=metadata.shape, dtype=metadata.dtype, data=metadata)
            f.attrs["metadata_n_frames"] = metadata[:, 0].max()
            f.attrs["metadata_fpath"] = str(metadata_path)

            # video
            f.attrs["video_fpath"] = str(video_path)
            f.attrs["video_n_frames"] = video_num_frames
            f.attrs["video_resolution"] = (video_width, video_height)
            f.attrs["video_fps"] = video_fps

        # make sure to close the video!
        cap.release()

        # create acoustic image json and dump
        ai_js = generate_acoustic_image_json(a_np, metadata, )

        this_file_res = {
            "videos": [
                {
                    "id": hdf_idx,
                    "file_name": clip_name.name,
                }
            ],
            "annotations": ai_js
        }

        # iterate over all masks
        for li in ai_js:
            # iterate over all polygons within the mask
            for poly in li["segmentation"]:
                # keep the largest pixel value within the mask
                poly_arr = np.array(poly)
                poly_amp = poly_arr[:, -1]
                pixel_amps.append(np.max(poly_amp))

        # dump the JSON
        js_path = outpath_with_split / clip_name.with_suffix(".json").name
        with open(js_path, "w") as f:
            json.dump(this_file_res, f, indent=4, ensure_ascii=False)

    # Compute summary statistics from all acoustic image JSONs and standardise
    pixel_arr = np.array(pixel_amps)
    starss_mu, starss_sd = np.mean(pixel_arr), np.std(pixel_arr)

    print(f"STARSS mean pixel amplitude: {starss_mu}")
    print(f"STARSS standard deviation pixel amplitude: {starss_sd}")

    # Standardise all the JSONs
    for js_path in outdir.rglob("**/*.json"):
        with open(js_path, "r") as js_in:
            js = json.load(js_in)

        new_res = []
        for obj_mask in js["annotations"]:
            std_seg = []
            for poly in obj_mask["segmentation"]:
                poly_arr = np.array(poly)
                poly_amp = poly_arr[:, -1]

                # Z-score
                poly_amp = (poly_amp - starss_mu) / starss_sd

                # Add 0.5 and clip
                poly_amp = np.clip(poly_amp + 0.5, 0.01, 1.0)

                # Create new array and replace the amplitude values with standardised version
                poly_new = poly_arr.copy()
                poly_new[:, -1] = poly_amp

                std_seg.append(poly_new.tolist())
            obj_mask["segmentation"] = std_seg
            new_res.append(obj_mask)
        js["annotations"] = new_res

        with open(js_path.with_name(js_path.stem + "_std").with_suffix(".json"), "w") as js_out:
            json.dump(js, js_out, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-src",
        type=str,
        help="Path to the source data directory",
        default=DEFAULT_EIGEN_DIRECTORY
    )
    parser.add_argument(
        "--outpath",
        type=str,
        help="Path to the output HDF5 dataset",
        default=DEFAULT_OUTPATH
    )

    # Parse arguments
    args = vars(parser.parse_args())

    main(**args)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create acoustic map using APGD for an audio file
"""

import math
import pandas as pd
import time
from argparse import ArgumentParser
from collections.abc import Sized, Iterable
from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.colors as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import mpl_toolkits.basemap as basemap
import numpy as np
import librosa
import skimage.util as skutil
import scipy.constants as constants
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg
import scipy.special as special
import scipy.signal.windows as windows
import scipy.io.wavfile as wavfile
import pyunlocbox as opt
from pyunlocbox.functions import dummy
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.animation import FuncAnimation

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
DEFAULT_OUTPATH = utils.get_project_root() / "outputs/eigen_dev_apgd_map"


def _deg2rad(coords_dict):
    """
    Take a dictionary with microphone array capsules and 3D polar coordinates to convert them from degrees to radians
    colatitude, azimuth, and radius (radius is left intact)
    """
    return {
        m: [math.radians(c[0]), math.radians(c[1]), c[2]]
        for m, c in coords_dict.items()
    }


def _polar2cart(coords_dict, units=None):
    """
    Take a dictionary with microphone array capsules and polar coordinates and convert to cartesian
    """
    if units is None or units != "degrees" and units != "radians":
        raise ValueError("you must specify units of 'degrees' or 'radians'")
    elif units == "degrees":
        coords_dict = _deg2rad(coords_dict)
    return {
        m: [
            c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0]),
        ]
        for m, c in coords_dict.items()
    }


def eq2cart(r, lat, lon):
    r = np.array([r]) #if chk.is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    return (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
        .to_cartesian()
        .xyz.to_value(u.dimensionless_unscaled)
    )


def pol2cart(r, colat, lon):
    lat = (np.pi / 2) - colat
    return eq2cart(r, lat, lon)


def get_xyz():
    mic_coords = _polar2cart(EIGENMIKE_COORDS, units='degrees')
    xyz = [[coo_ for coo_ in mic_coords[ch]] for ch in mic_coords]
    return xyz


def spherical_jn_series_threshold(x, table_lookup=True, epsilon=1e-2):
    r"""
    Convergence threshold of series :math:`f_{n}(x) = \sum_{q = 0}^{n} (2 q + 1) j_{q}^{2}(x)`.
    """
    if not (0 < epsilon < 1):
        raise ValueError("Parameter[epsilon] must lie in (0, 1).")

    if table_lookup:
        abs_path = utils.get_project_root() / "starss_representations/audio/spherical_jn_series_threshold.csv"
        data = pd.read_csv(abs_path).sort_values(by="x")

        x = np.abs(x)
        idx = int(np.digitize(x, bins=data["x"].values))
        if idx == 0:  # Below smallest known x.
            n = data["n_threshold"].iloc[0]
        else:
            if idx == len(data):  # Above largest known x.
                ratio = data["n_threshold"].iloc[-1] / data["x"].iloc[-1]
            else:
                ratio = data["n_threshold"].iloc[idx - 1] / data["x"].iloc[idx - 1]
            n = int(np.ceil(ratio * x))

        return n
    else:

        def series(n_, x_):
            q = np.arange(n_)
            _2q1 = 2 * q + 1
            _sph = special.spherical_jn(q, x_) ** 2

            return np.sum(_2q1 * _sph)

        n_opt = int(0.95 * x)
        while True:
            n_opt += 1
            if 1 - series(n_opt, x) < epsilon:
                return n_opt


def nyquist_rate(xyz, wl):
    """
    Order of imageable complex plane-waves by an instrument.
    """
    baseline = linalg.norm(xyz[:, np.newaxis, :] - xyz[:, :, np.newaxis], axis=0)
    return spherical_jn_series_threshold((2 * np.pi / wl) * baseline.max())


def fibonacci(n, direction=None, fo_v=None):
    # This is the type of tesselation that we are using

    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)

        if fo_v is not None:
            if not (0 < np.rad2deg(fo_v) < 360):
                raise ValueError("Parameter[FoV] must be in (0, 360) degrees.")
        else:
            raise ValueError("Parameter[FoV] must be specified if Parameter[direction] provided.")

    if n < 0:
        raise ValueError("Parameter[N] must be non-negative.")

    n_px = 4 * (n + 1) ** 2
    n = np.arange(n_px)

    colat = np.arccos(1 - (2 * n + 1) / n_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    xyz = np.stack(pol2cart(1, colat, lon), axis=0)

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


def get_field():
    # TODO: this is what we need to mess with
    sh_order = 10
    r = fibonacci(sh_order)
    r_mask = np.abs(r[2, :]) < np.sin(np.deg2rad(90))
    r = r[:, r_mask]  # Shrink visible view to avoid border effects.
    # this is cartesian coordinates: (3, n_px)
    return r


def steering_operator(xyz, r):
    """
    Steering matrix.
    """
    freq, bw = np.linspace(50, 4500, 9), 50.0  # [Hz]
    wl = constants.speed_of_sound / (freq.max() + 500)
    if wl <= 0:
        raise ValueError("Parameter[wl] must be positive.")

    scale = 2 * np.pi / wl
    return np.exp((-1j * scale * xyz.T) @ r)


def extract_visibilities(_data, _rate, t, fc, bw, alpha):
    """
    Transform time-series to visibility matrices.
    """
    n_stft_sample = int(_rate * t)
    if n_stft_sample == 0:
        raise ValueError('Not enough samples per time frame.')
    # print(f'Samples per STFT: {N_stft_sample}')

    n_sample = (_data.shape[0] // n_stft_sample) * n_stft_sample
    n_channel = _data.shape[1]
    stf_data = (skutil.view_as_blocks(_data[:n_sample], (n_stft_sample, n_channel))
                .squeeze(axis=1))  # (n_stf, N_stft_sample, n_channel)

    window = windows.tukey(M=n_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
    stf_win_data = stf_data * window  # (n_stf, N_stft_sample, n_channel)
    n_stf = stf_win_data.shape[0]

    stft_data = np.fft.fft(stf_win_data, axis=1)  # (n_stf, N_stft_sample, n_channel)
    # Find frequency channels to average together.
    idx_start = int((fc - 0.5 * bw) * n_stft_sample / _rate)
    idx_end = int((fc + 0.5 * bw) * n_stft_sample / _rate)
    collapsed_spectrum = np.sum(stft_data[:, idx_start:idx_end + 1, :], axis=1)

    # Don't understand yet why conj() on first term?
    # collapsed_spectrum = collapsed_spectrum[0,:]
    return (
        collapsed_spectrum.reshape(n_stf, -1, 1).conj() *
        collapsed_spectrum.reshape(n_stf, 1, -1)
    )


# noinspection PyArgumentList
def eigh_max(a):
    r"""
    Evaluate :math:`\mu_{\max}(\bbB)` with

    :math:
    B = (\overline{\bbA} \circ \bbA)^{H} (\overline{\bbA} \circ \bbA)
    """
    if a.ndim != 2:
        raise ValueError('Parameter[A] has wrong dimensions.')

    def matvec(v):
        v = v.reshape(-1)

        c = (a * v) @ a.conj().T
        d = c @ a
        return np.sum(a.conj() * d, axis=0).real

    m, n = a.shape
    b = splinalg.LinearOperator(shape=(n, n), matvec=matvec, dtype=np.float64)
    d_max = splinalg.eigsh(b, k=1, which='LM', return_eigenvectors=False)
    return d_max[0]


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

    def _update_step(self, solver, objective, niter):
        """
        Update the step size for the next iteration.
        """
        return self._step

    def _update_sol(self, solver, objective, niter):
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
            err = np.linalg.norm(solver.sol - last_sol)    # noqa
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
        print('    objective function f(sol) = {:e}'.format(current))    # noqa
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
        skutil.view_as_windows(
            s_sti,
            window_shape=(n_sti_per_stationary_block, n_channel, n_channel),
            step=(n_sti_per_stationary_block, n_channel, n_channel)
        )
        .squeeze(axis=(1, 2))
        .sum(axis=1)
    )


# noinspection PyUnresolvedReferences
def wrapped_rad2deg(lat_r: np.ndarray, lon_r: np.ndarray) -> tuple:
    """
    Equatorial coordinate [rad] -> [deg] unit conversion.
    Output longitude guaranteed to lie in [-180, 180) [deg].
    """
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d


# noinspection PyUnresolvedReferences
def cart2pol(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Cartesian coordinates to Polar coordinates.
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)

    return r, colat, lon


def cart2eq(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Cartesian coordinates to Equatorial coordinates.
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (np.pi / 2) - colat
    return r, lat, lon


def cmap_from_list(name: list, colors: list, n: int = 256, gamma: float = 1.0):
    """
    Create segmented colormap from list of colors.
    """

    if not isinstance(colors, Iterable):
        raise ValueError('colors must be iterable')

    # List of value, color pairs
    if (
        isinstance(colors[0], Sized) and
        (len(colors[0]) == 2) and
        (not isinstance(colors[0], str))
    ):
        vals, colors = zip(*colors)
    else:
        vals = np.linspace(0, 1, len(colors))

    cdict = dict(red=[], green=[], blue=[], alpha=[])
    for val, color in zip(vals, colors):
        r, g, b, a = cm.to_rgba(color)
        cdict['red'].append((val, r, r))
        cdict['green'].append((val, g, g))
        cdict['blue'].append((val, b, b))
        cdict['alpha'].append((val, a, a))

    return cm.LinearSegmentedColormap(name, cdict, n, gamma)


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance
    """
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        u_, s, vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(u_[1, 0], u_[0, 0]))
        width, height = 2 * np.sqrt(s)
        print(width, height)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ell =  mpatches.Ellipse(
            position,
            nsig * width,
            nsig * height,
            angle,
            **kwargs
        )
        print("Width", ell.width)
        ax.add_patch(ell)


def plot_gmm(
        gmm: GaussianMixture,
        x: np.ndarray,
        label: bool = True,
        ax: plt.Axes = None
) -> None:
    """
    Plot gaussian mixture model
    """
    ax = ax or plt.gca()
    labels = gmm.fit(x).predict(x)    # noqa
    if label:
        ax.scatter(x[:, 0], x[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(x[:, 0], x[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def draw_map(
        i: np.ndarray,
        r: np.ndarray,
        lon_ticks: np.ndarray,
        show_labels: bool = False,
        show_axis: bool = False,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        kmeans: bool = False,
        gaussian_mixture: bool = False,
        az0: list[float] | float = None,
        el0: list[float] | float = None,
        mask_radius: float = 30,
        alpha: float = 0.9
) -> tuple:

    """
    Draw acoustic map with optional angular masking.
    """

    # Convert coordinate grid to elevation/azimuth (already spherical)
    _, r_el, r_az = cart2eq(*r)
    r_el, r_az = wrapped_rad2deg(r_el, r_az)
    r_el_min, r_el_max = np.around([np.min(r_el), np.max(r_el)])
    r_az_min, r_az_max = np.around([np.min(r_az), np.max(r_az)])

    # Save original layout
    orig_pos = ax.get_position()
    orig_aspect = ax.get_aspect()

    # Basemap init
    bm = basemap.Basemap(
        projection='mill',
        llcrnrlat=r_el_min,
        urcrnrlat=r_el_max,
        llcrnrlon=r_az_min,
        urcrnrlon=r_az_max,
        resolution='c',
        ax=ax
    )

    if show_axis:
        bm_labels = [1, 0, 0, 1]
        bm.drawparallels(
            np.linspace(r_el_min, r_el_max, 5),
            color='w',
            dashes=[1, 0],
            labels=bm_labels,
            labelstyle='+/-',
            textcolor='#565656',
            zorder=0,
            linewidth=2
        )
        bm.drawmeridians(
            lon_ticks,
            color='w',
            dashes=[1, 0],
            labels=bm_labels,
            labelstyle='+/-',
            textcolor='#565656',
            zorder=0,
            linewidth=2
        )

    if show_labels:
        ax.set_xlabel('Azimuth (degrees)', labelpad=20)
        ax.set_ylabel('Elevation (degrees)', labelpad=40)

    # Project spherical to 2D basemap coordinates
    r_x, r_y = bm(r_az, r_el)

    # optional masking
    if az0 is not None and el0 is not None:
        if isinstance(az0, float):
            az0 = np.ndarray([az0])
        if isinstance(el0, float):
            el0 = np.ndarray([el0])

        # Convert to radians
        az1 = np.deg2rad(r_az)
        el1 = np.deg2rad(r_el)
        az0r = np.deg2rad(az0)
        el0r = np.deg2rad(el0)

        # Spherical angular distance
        ang = np.arccos(
            np.sin(el0r[:, None]) * np.sin(el1[None, :]) +
            np.cos(el0r[:, None]) * np.cos(el1[None, :]) * np.cos(az1[None, :] - az0r[:, None])
        )
        ang_deg = np.rad2deg(ang)

        # compute mask, use logical OR across rows
        mask = (ang_deg <= mask_radius).any(axis=0)

        # apply mask
        i[:, ~mask] = 0

    # triangulation + plotting
    triangulation = tri.Triangulation(r_x, r_y)

    n_px = i.shape[1]
    mycmap = cmap_from_list('mycmap', i.T, n=n_px)
    colors_cmap = np.arange(n_px)

    ax.tripcolor(
        triangulation,
        colors_cmap,
        cmap=mycmap,
        shading='gouraud',  # removes interpolation patterns
        edgecolors='none',
        linewidth=0,
        alpha=alpha,
        zorder=100,
    )

    # annotate the target point (always)
    if az0 is not None and el0 is not None:
        for az0_, el0_ in zip(az0, el0):
            px, py = bm(az0_, el0_)
            ax.plot(px, py, 'ro', markersize=10, zorder=1000)
            ax.text(px, py, f" ({az0_}°, {el0_}°)", color="red", fontsize=9, zorder=1000)

    # clustering with kmeans/GMM if required
    cluster_center = None
    if kmeans:
        npts = 18  # find N maximum points
        i_s = np.square(i).sum(axis=0)
        max_idx = i_s.argsort()[-npts:][::-1]
        x_y = np.column_stack((r_x[max_idx], r_y[max_idx]))  # stack N max energy points
        km_res = KMeans(n_clusters=3).fit(x_y)  # apply k-means to max points
        # get center of the cluster of N points
        clusters = km_res.cluster_centers_    # noqa
        ax.scatter(r_x[max_idx], r_y[max_idx], c='b', s=5)  # plot all N points
        ax.scatter(clusters[:, 0], clusters[:, 1], s=500, alpha=0.3)  # plot the center as a large point
        cluster_center = bm(clusters[:, 0][0], clusters[:, 1][0], inverse=True)

    elif gaussian_mixture:
        npts = 18
        i_s = np.square(i).sum(axis=0)
        max_idx = i_s.argsort()[-npts:][::-1]
        x_y = np.column_stack((r_x[max_idx], r_y[max_idx]))  # stack N max energy points
        gmm = GaussianMixture(n_components=1, random_state=42)
        plot_gmm(gmm, x_y)

    # cosmetic cleanup
    ax.tick_params(left=False, right=False, top=False, bottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for txt in ax.texts:
        txt.set_visible(False)

    # restore axis positions
    ax.set_position(orig_pos)
    ax.set(xticklabels=[], yticklabels=[])
    ax.set_aspect(orig_aspect)

    return fig, ax, cluster_center


def to_rgb(i: np.ndarray) -> np.ndarray:
    """
    Convert real-valued intensity array (per frequency) with shape (N_band, N_px) to color-band array (3, N_px)

    If N_band > 9, set N_band == 9
    """
    n_px = i.shape[1]
    # Create a copy of the input array if it's going to be modified
    i_copy = i.copy()
    if i.shape[0] != 9:
        i_copy = i_copy[1:10, :]  # grab 9 frequency bands
    # Reshape and sum to get (3, N_px)
    return i_copy.reshape((3, 3, n_px)).sum(axis=1)


def get_visibility_matrix(
    audio_in: np.ndarray,
    fs: int,
    apgd: bool = False,
    t_sti: float = 10e-3,
    scale: str = "linear",
    nbands: int = 9,
    frame_cap: int = None
) -> tuple:
    """
    Compute visibility matrix from audio data.
    """

    print("\n=== get_visibility_matrix START ===")
    print(f"Sampling rate: {fs}")
    print(f"APGD enabled: {apgd}")
    print(f"t_sti: {t_sti}")
    print(f"Scale: {scale}, nbands: {nbands}")

    # --- Frequency band creation ---
    print("[1/6] Computing frequency bands...")
    # Use spacing between 50 and 4500 Hz as in LAM paper
    if scale == "linear":
        freq = np.linspace(50, 4500, nbands)
        bw = 50.0
    elif scale == "log":
        freq = librosa.mel_frequencies(n_mels=nbands, fmin=50, fmax=4500)
        bw = 50
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

    visibilities = []
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

        visibilities_per_frame = []
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

            visibilities_per_frame.append(s_norm)

            if apgd:
                i_apgd = solve(
                    s_norm,
                    a,
                    gamma=apgd_gamma,
                    x0=i_prev.copy(),
                    verbosity='NONE'
                )
                apgd_per_band[s_idx] = i_apgd['sol']
                i_prev = i_apgd['sol']
            else:
                apgd_per_band[s_idx] = [0]

        apgd_map.append(apgd_per_band)
        visibilities.append(visibilities_per_frame)

    print("\n[6/6] Final assembly...")
    vis_arr = np.array(visibilities)
    # bands, frames, number of interpolated pixels -> need the tesselation
    apgd_arr = np.array(apgd_map)

    print(f" → Visibility array shape: {vis_arr.shape}")
    print(f" → APGD map shape: {apgd_arr.shape}")
    print("=== get_visibility_matrix END ===\n")

    return vis_arr, apgd_arr, r


def acoustic_map_to_rgb(apgd_arr: np.ndarray, normalize: bool = True, scaling: float = None) -> np.ndarray:
    """
    Convert intensity map with shape (n_bands, n_frames, n_px) to shape (3, n_frames)

    If normalise, scale so that maximum intensity across all frames == 1
    If scaling, multiply all values by a constant then clip to within original range
    """
    ip = np.zeros((apgd_arr.shape[1], 3, apgd_arr.shape[-1]))
    for fi in range(apgd_arr.shape[1]):
        vs__ = apgd_arr[:, fi, :]
        map_vs__ = to_rgb(vs__)
        ip[fi, :, :] = map_vs__

    # Normalize or not
    if normalize:
        ip /= ip.max()

    # Scale by a floating point value then clip to avoid overflow
    if scaling:
        orig_min, orig_max = ip.min(), ip.max()
        ip *= scaling
        ip = np.clip(ip, orig_min, orig_max)

    return ip

def generate_acoustic_map_video(
        apgd_arr: np.ndarray,
        r: np.ndarray,
        ts: float,
        f_out: str | Path = None,
        fig: plt.Figure = None,
        ax: plt.Figure = None
) -> FuncAnimation:
    """
    Generate acoustic map as video
    """

    def update(frame_idx: int) -> plt.Axes:
        # Clear the current canvas
        ax.clear()

        # Get the current frame from the normalised RGB array
        vs = ip_norm[frame_idx, :, :]

        # Redraw frame
        draw_map(
            r=r,
            i=vs,
            lon_ticks=np.linspace(-180, 180, 5),
            fig=fig,
            ax=ax,
            show_labels=False,
            show_axis=True
        )

        # Set plot aesthetics
        ax.set(xticks=[], yticks=[], title="Acoustic Map", xticklabels=[], yticklabels=[])
        ax.invert_xaxis()
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        return ax

    # Convert intensity map to RGB
    ip_norm = acoustic_map_to_rgb(apgd_arr, normalize=True)

    # Create figure and axis if not already existing
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    # Need to plot the first frame
    _ = update(0)

    # Create the animation and save if required
    anim = FuncAnimation(
        fig,
        update,
        frames=apgd_arr.shape[1],
        interval=ts,
        repeat=False
    )
    if f_out is not None:
        anim.save(f_out)

    return anim


def main(data_src: str, outpath: str) -> None:
    eigenmike_files = [p for p in Path(data_src).rglob("**/*.wav") if "._" not in str(p)]

    # Sanitise output directory
    outdir = Path(outpath)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # logarithmically space duration from 2sec to 10ms, rounded to 3 dp
    t_logs_spaced = np.logspace(np.log10(1e-3), np.log10(0.2), 9)
    t_logs_spaced = np.round(t_logs_spaced, 3)[::-1]

    # Iterate over every audio file
    for clip_name in tqdm(eigenmike_files, desc="Processing files..."):

        # Load in the WAV file
        sr, eigen_sig = wavfile.read(clip_name)

        # Iterate over every timescale
        for t_sti in t_logs_spaced:

            # Set filepath for this timescale + clip
            file_outpath = f"{outdir}/{str(clip_name).split('/')[-1].split('.')[0]}_{round(t_sti * 10000)}ms.mp4"

            # compute visibility graph matrix (32ch)
            # TODO: implement upsampling from 4ch -> 32ch
            vsg_sig, apgd, mapper = get_visibility_matrix(
                eigen_sig,
                sr,
                apgd=True,
                t_sti=t_sti,
                scale="linear",
                nbands=9
            )

            generate_acoustic_map_video(
                apgd_arr=apgd,
                r=mapper,
                # Need to map to milliseconds for mpl
                ts=t_sti * 10000,
                f_out=file_outpath
            )


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

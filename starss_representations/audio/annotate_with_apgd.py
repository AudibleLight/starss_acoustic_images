import math
import pandas as pd
import random
import time
from argparse import ArgumentParser
from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
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
DEFAULT_OUTPATH = utils.get_project_root() / "outputs/eigen_dev/apgd.hdf"


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

    return xyz


def get_field():
    sh_order = 10
    r = fibonacci(sh_order)
    r_mask = np.abs(r[2, :]) < np.sin(np.deg2rad(90))
    r = r[:, r_mask]  # Shrink visible view to avoid border effects.
    return r


def steering_operator(xyz, r):
    """
    Steering matrix.
    """
    freq, bw = (skutil  # Center frequencies to form images
            .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
            .mean(axis=-1)), 50.0  # [Hz]
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
            err = np.linalg.norm(solver.sol - last_sol)
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
        print('    objective function f(sol) = {:e}'.format(current))
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


def get_visibility_matrix(audio_in, fs, apgd=False, t_sti=10e-3, scale="linear", nbands=9):

    print("\n=== get_visibility_matrix START ===")
    print(f"Sampling rate: {fs}")
    print(f"APGD enabled: {apgd}")
    print(f"t_sti: {t_sti}")
    print(f"Scale: {scale}, nbands: {nbands}")

    # --- Frequency band creation ---
    print("[1/6] Computing frequency bands...")
    if scale == "linear":
        freq, bw = (skutil
                    .view_as_windows(np.linspace(1500, 4500, nbands), (2,), 1)
                    .mean(axis=-1)), 50.0
    elif scale == "log":
        freq, bw = librosa.mel_frequencies(n_mels=nbands, fmin=50, fmax=4500), 50
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
        print(f"\n--- Band {i+1}/{nbands} | freq={freq[i]:.2f} Hz ---")

        t_stationarity = 10 * t_sti
        print("    → Computing visibility matrices...")
        s = form_visibility(audio_in, fs, freq[i], bw, t_sti, t_stationarity)
        n_sample = s.shape[0]
        print(f"    → Visibility frames: {n_sample}")

        visibilities_per_frame = []
        apgd_gamma = 0.5
        apgd_per_band = np.zeros((n_sample, n_px))
        i_prev = np.zeros((n_px,))

        print("    → Processing frames...")
        for s_idx in range(n_sample):

            if s_idx % 20 == 0 or s_idx == n_sample - 1:
                print(f"        Frame {s_idx+1}/{n_sample}")

            # Eigen-decomposition
            s_d, s_v = linalg.eigh(s[s_idx])
            if s_d.max() <= 0:
                s_d[:] = 0
            else:
                s_d = np.clip(s_d / s_d.max(), 0, None)
            s_norm = (s_v * s_d) @ s_v.conj().T

            visibilities_per_frame.append(s_norm)

            if apgd:
                i_apgd = solve(s_norm, a, gamma=apgd_gamma, x0=i_prev.copy(),
                               verbosity='NONE')
                apgd_per_band[s_idx] = i_apgd['sol']
                i_prev = i_apgd['sol']
            else:
                apgd_per_band[s_idx] = [0]

        apgd_map.append(apgd_per_band)
        visibilities.append(visibilities_per_frame)

    print("\n[6/6] Final assembly...")
    vis_arr = np.array(visibilities)
    apgd_arr = np.array(apgd_map)

    print(f" → Visibility array shape: {vis_arr.shape}")
    print(f" → APGD map shape: {apgd_arr.shape}")
    print("=== get_visibility_matrix END ===\n")

    return vis_arr, apgd_arr


def main(data_src: str, outpath: str) -> None:
    eigenmike_files = [p for p in Path(data_src).rglob("**/*.wav") if "._" not in str(p)]

    # Sanitise output directory
    outdir = Path(outpath).parent
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # logarithmically space duration from 10ms to 2sec, rounded to 3 dp
    t_logs_spaced = np.logspace(np.log10(20e-3), np.log10(200e-3),9)
    t_logs_spaced = np.round(t_logs_spaced, 3)

    vg_labels = []
    apgd_labels = []
    t_sti_list = []

    with File(outpath, "w") as f:
        for clip_name in tqdm(eigenmike_files, desc="Processing files..."):
            t_sti = random.choice(t_logs_spaced)
            sr, eigen_sig = wavfile.read(clip_name)

            # visibility graph matrix 32ch
            vsg_sig, apgd = get_visibility_matrix(
                eigen_sig,
                sr,
                apgd=True,
                t_sti=t_sti,
                scale="log",
                nbands=16
            )

            # (nframes, nbands, nch, nch)
            vg_labels.append(vsg_sig.transpose(1, 0, 2, 3))

            # (nframes, nbands, Npx)
            apgd_labels.append(apgd.transpose(1, 0, 2))

            for _ in range(apgd.shape[1]):
                t_sti_list.append(t_sti)

        b_np = np.vstack(vg_labels)
        c_np = np.vstack(apgd_labels)
        d_np = np.vstack(t_sti_list)

        print("shape of b_np", b_np.shape)
        print("shape of c_np", c_np.shape)
        print("shape of d_np", d_np.shape)

        f.create_dataset("em32", shape=b_np.shape, dtype=b_np.dtype, data=b_np)
        f.create_dataset("apgd", shape=c_np.shape, dtype=c_np.dtype, data=c_np)
        f.create_dataset("dur", shape=d_np.shape, dtype=d_np.dtype, data=d_np)
        f.attrs["sr"] = sr


if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser(description="Generate HDF5 dataset for LAM training.")
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

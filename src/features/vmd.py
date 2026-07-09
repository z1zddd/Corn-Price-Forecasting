"""VMD feature wrapper around vmdpy's copied implementation."""

from __future__ import annotations

import numpy as np


def VMD(f, alpha, tau, K, DC, init, tol):
    """Copied from vrcarva/vmdpy with no algorithmic changes."""

    if len(f) % 2:
        f = f[:-1]
    fs = 1.0 / len(f)
    ltemp = len(f) // 2
    fMirr = np.append(np.flip(f[:ltemp], axis=0), f)
    fMirr = np.append(fMirr, np.flip(f[-ltemp:], axis=0))
    T = len(fMirr)
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - (1 / T)
    Niter = 500
    Alpha = alpha * np.ones(K)
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[: T // 2] = 0
    omega_plus = np.zeros([Niter, K])
    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K)))
    else:
        omega_plus[0, :] = 0
    if DC:
        omega_plus[0, 0] = 0
    lambda_hat = np.zeros([Niter, len(freqs)], dtype=complex)
    uDiff = tol + np.spacing(1)
    n = 0
    sum_uk = 0
    u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=complex)
    while uDiff > tol and n < Niter - 1:
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]
        u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
            1.0 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2
        )
        if not DC:
            omega_plus[n + 1, k] = np.dot(freqs[T // 2 : T], (abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2
            )
        for k in np.arange(1, K):
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2
            )
            omega_plus[n + 1, k] = np.dot(freqs[T // 2 : T], (abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2
            )
        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)
        n = n + 1
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1 / T) * np.dot(
                (u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                np.conj((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])),
            )
        uDiff = np.abs(uDiff)
    Niter = np.min([Niter, n])
    omega = omega_plus[:Niter, :]
    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)
    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T // 2 : T, :] = u_hat_plus[Niter - 1, T // 2 : T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[Niter - 1, T // 2 : T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])
    u = np.zeros([K, len(t)])
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))
    u = u[:, T // 4 : 3 * T // 4]
    u_hat = np.zeros([u.shape[1], K], dtype=complex)
    for k in range(K):
        u_hat[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))
    return u, u_hat, omega


def vmd_modes(series, alpha: float = 2000, tau: float = 0.0, k: int = 3, dc: int = 0, init: int = 1, tol: float = 1e-7):
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("VMD expects a 1D series.")
    return VMD(values, alpha=alpha, tau=tau, K=k, DC=dc, init=init, tol=tol)[0]


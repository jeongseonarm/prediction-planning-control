"""
Vectorized Frenet -> Cartesian conversion.

Same formulas as CartesianFrenetConverter.frenet_to_cartesian (PythonRobotics,
Wang Zheng, MIT licence; Werling et al. 2010), evaluated for whole arrays at
once. The scalar original is left untouched.
"""

import numpy as np


def _normalize_angle(angle):
    a = np.fmod(angle + np.pi, 2.0 * np.pi)
    a = np.where(a < 0.0, a + 2.0 * np.pi, a)
    return a - np.pi


def frenet_to_cartesian_batch(
    rx, ry, rtheta, rkappa, rdkappa, s_dot, s_ddot, d, d_prime, d_pprime
):
    """
    All arguments are arrays of the same shape, (n,) or (G, n) (reference pose
    at s, and the Frenet state s'(t), s''(t), d(s), d'(s), d''(s)).
    Returns an array of shape (..., 6) with columns x, y, theta, kappa, v, a.
    """
    cos_r = np.cos(rtheta)
    sin_r = np.sin(rtheta)

    x = rx - sin_r * d
    y = ry + cos_r * d

    one_minus_kappa_r_d = 1 - rkappa * d

    tan_delta = d_prime / one_minus_kappa_r_d
    delta = np.arctan2(d_prime, one_minus_kappa_r_d)
    cos_delta = np.cos(delta)

    theta = _normalize_angle(delta + rtheta)

    kappa_r_d_prime = rdkappa * d + rkappa * d_prime

    kappa = (
        ((d_pprime + kappa_r_d_prime * tan_delta) * cos_delta * cos_delta)
        / one_minus_kappa_r_d + rkappa
    ) * cos_delta / one_minus_kappa_r_d

    d_dot = d_prime * s_dot
    v = np.sqrt(
        one_minus_kappa_r_d * one_minus_kappa_r_d * s_dot * s_dot + d_dot * d_dot
    )

    delta_theta_prime = one_minus_kappa_r_d / cos_delta * kappa - rkappa

    a = (
        s_ddot * one_minus_kappa_r_d / cos_delta
        + s_dot * s_dot / cos_delta * (d_prime * delta_theta_prime - kappa_r_d_prime)
    )

    return np.stack((x, y, theta, kappa, v, a), axis=-1)

"""
Smoothing-spline reference path builder.

Fits a quintic smoothing spline to discrete waypoints (e.g. a Hybrid A* path)
instead of interpolating them exactly, so that the curvature (kappa) and its
derivative (dkappa = d kappa / ds) stay smooth.

Returns
    reference : (N, 6) array [s, x, y, yaw, kappa, dkappa], s = arc length
    info      : dict with the fitted spline (tck) and deviation diagnostics

A smoothing spline has no end conditions and tends to bend at both ends, so the
path is extended with straight segments before fitting and only the original
span is returned.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


def _drop_close_points(xy, min_dist):
    """Drop consecutive points closer than min_dist (the last point is kept)."""
    keep = [0]
    for i in range(1, len(xy)):
        if np.linalg.norm(xy[i] - xy[keep[-1]]) >= min_dist:
            keep.append(i)

    if keep[-1] != len(xy) - 1:
        keep[-1] = len(xy) - 1

    return np.asarray(keep, dtype=int)


def _end_tangent_from_points(xy, u, at_start, window=3.0):
    """Unit heading at an end point, from a polynomial fit of nearby points."""
    if at_start:
        mask = u <= u[0] + window
        uu = u[mask] - u[0]
    else:
        mask = u >= u[-1] - window
        uu = u[mask] - u[-1]

    pts = xy[mask]
    deg = 2 if len(uu) >= 4 else 1

    cx = np.polyfit(uu, pts[:, 0], deg)
    cy = np.polyfit(uu, pts[:, 1], deg)

    t = np.array([cx[-2], cy[-2]])  # first derivative at u = 0
    return t / np.linalg.norm(t)


def _end_tangent(xy, u, yaw, at_start):
    """Use the given yaw if it agrees with the points, else estimate from them."""
    t_pts = _end_tangent_from_points(xy, u, at_start)

    if yaw is None or not np.isfinite(yaw):
        return t_pts

    t_yaw = np.array([np.cos(yaw), np.sin(yaw)])

    if float(np.dot(t_yaw, t_pts)) > 0.5:  # within about 60 deg
        return t_yaw

    return t_pts


def build_reference_path(
    global_path,
    ds=0.1,
    sigma=0.1,
    order=5,
    end_extension=4.0,
    use_given_yaw=True,
    fine_ds=0.01,
    min_point_distance=1e-3,
):
    """
    Parameters
    ----------
    global_path : (N, >=2) array
        [x, y, (yaw)] waypoints. The yaw column is only used for end headings.
    ds : float
        Output spacing along the path [m].
    sigma : float
        Allowed RMS deviation between waypoints and the spline [m].
        Smaller follows the waypoints more closely (kappa oscillates more),
        larger is smoother. Keep it at or above the waypoint noise level,
        otherwise kappa can blow up.
    order : int
        Spline degree. 5 makes dkappa continuous.
    end_extension : float
        Length of the straight extension added at each end before fitting [m].
    use_given_yaw : bool
        Use the yaw column (rad) for the end headings if available.
    fine_ds : float
        Internal sampling step used for the arc-length reparameterization [m].
    min_point_distance : float
        Consecutive waypoints closer than this are dropped [m].
    """
    global_path = np.asarray(global_path, dtype=float)

    if global_path.ndim != 2 or global_path.shape[1] < 2:
        raise ValueError("global_path must have shape (N, >=2).")

    if ds <= 0 or sigma <= 0 or fine_ds <= 0 or end_extension < 0:
        raise ValueError(
            "ds, sigma, fine_ds must be positive and end_extension >= 0."
        )

    if not np.all(np.isfinite(global_path[:, :2])):
        raise ValueError("global_path contains non-finite coordinates.")

    # 1. Clean waypoints
    keep = _drop_close_points(global_path[:, :2], min_point_distance)
    xy = global_path[keep, :2]

    yaw_start = yaw_end = None
    if use_given_yaw and global_path.shape[1] >= 3:
        yaw_start = float(global_path[keep[0], 2])
        yaw_end = float(global_path[keep[-1], 2])

    if len(xy) < 4:
        raise ValueError("At least four distinct waypoints are required.")

    seg = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)

    # A turn of more than 90 deg between segments is treated as a cusp.
    cos_turn = np.sum(seg[:-1] * seg[1:], axis=1) / (
        seg_len[:-1] * seg_len[1:]
    )
    if np.any(cos_turn < 0.0):
        idx = int(np.argmax(cos_turn < 0.0))
        raise ValueError(
            f"Heading changes by more than 90 deg near waypoint {idx + 1} "
            "(cusp / direction switch). Split the path at cusps."
        )

    u_orig = np.concatenate(([0.0], np.cumsum(seg_len)))
    m = len(xy)

    # 2. Straight extensions at both ends
    if end_extension > 0.0:
        step = float(np.median(seg_len))
        n_ext = max(int(np.ceil(end_extension / step)), 1)
        offsets = step * np.arange(1, n_ext + 1)[:, None]

        t0 = _end_tangent(xy, u_orig, yaw_start, at_start=True)
        t1 = _end_tangent(xy, u_orig, yaw_end, at_start=False)

        pre = xy[0] - offsets[::-1] * t0
        post = xy[-1] + offsets * t1

        xy_fit = np.vstack((pre, xy, post))
        idx_start = n_ext
        idx_end = n_ext + m - 1
    else:
        xy_fit = xy
        idx_start = 0
        idx_end = m - 1

    # 3. Smoothing spline fit, parameterized by cumulative chord length
    seg_fit = np.linalg.norm(np.diff(xy_fit, axis=0), axis=1)
    u = np.concatenate(([0.0], np.cumsum(seg_fit)))
    n_fit = len(xy_fit)
    k = int(min(order, n_fit - 1))

    # sum((w * residual)^2) <= s with w = 1 / sigma and s = n gives RMS ~ sigma
    tck, _ = splprep(
        [xy_fit[:, 0], xy_fit[:, 1]],
        w=np.full(n_fit, 1.0 / sigma),
        u=u,
        k=k,
        s=float(n_fit),
    )

    u_start = float(u[idx_start])
    u_end = float(u[idx_end])

    # 4. Arc-length reparameterization of the original span
    u_fine = np.arange(u_start, u_end, fine_ds)
    if u_fine[-1] < u_end:
        u_fine = np.append(u_fine, u_end)

    fdx, fdy = splev(u_fine, tck, der=1)
    speed = np.hypot(fdx, fdy)
    arc = np.concatenate((
        [0.0],
        np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u_fine)),
    ))
    total_length = float(arc[-1])

    s_grid = np.arange(0.0, total_length, ds)
    if total_length - s_grid[-1] > 1e-6:
        s_grid = np.append(s_grid, total_length)
    else:
        s_grid[-1] = total_length

    u_s = np.interp(s_grid, arc, u_fine)

    # 5. Pose, curvature and curvature rate (derivatives w.r.t. arc length s)
    x, y = splev(u_s, tck, der=0)
    dx, dy = splev(u_s, tck, der=1)
    ddx, ddy = splev(u_s, tck, der=2)
    dddx, dddy = splev(u_s, tck, der=3)

    a = dx * ddy - dy * ddx
    b = dx * dddy - dy * dddx
    c = dx * ddx + dy * ddy
    d = dx * dx + dy * dy

    kappa = a / d**1.5
    dkappa = (b * d - 3.0 * a * c) / d**3
    yaw = np.unwrap(np.arctan2(dy, dx))

    # 6. Deviation of the spline from the original waypoints
    fx, fy = splev(u_fine, tck, der=0)
    tree = cKDTree(np.column_stack((fx, fy)))
    deviation, _ = tree.query(xy)

    reference = np.column_stack((s_grid, x, y, yaw, kappa, dkappa))

    info = {
        "tck": tck,
        "sigma": float(sigma),
        "order": k,
        "end_extension": float(end_extension),
        "n_waypoints": int(len(global_path)),
        "n_used_waypoints": int(m),
        "total_length": total_length,
        "max_deviation": float(np.max(deviation)),
        "rms_deviation": float(np.sqrt(np.mean(deviation**2))),
    }

    return reference, info

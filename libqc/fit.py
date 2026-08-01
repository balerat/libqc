import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from libqc.misc import generate_xy_list


def cloud_rms(psi, dx):
    """Density-weighted RMS radius about the centroid. psi = |ψ|² (>=0)."""
    x, y = generate_xy_list(psi)  # pixel grid
    x = x * dx
    y = y * dx
    tot = psi.sum()
    x0 = (psi * x).sum() / tot  # centroid (don't assume centered)
    y0 = (psi * y).sum() / tot
    sx = np.sqrt((psi * (x - x0) ** 2).sum() / tot)
    sy = np.sqrt((psi * (y - y0) ** 2).sum() / tot)
    r_rms = np.sqrt(sx**2 + sy**2)  # = sqrt(<r²> - <r>²)
    return sx, sy, r_rms


def fit_width(density, dx, smooth_px=4.0):
    """Gaussian-envelope fit of a 2D density. Returns (sigma_x, sigma_y) in dx units."""
    env = gaussian_filter(density, smooth_px)  # blur lattice spikes, keep envelope
    x, y = generate_xy_list(env)
    x = x * dx
    y = y * dx
    sx0, sy0, _ = cloud_rms(env, dx)  # seed from moments
    tot = env.sum()
    p0 = [env.max(), (env * x).sum() / tot, (env * y).sum() / tot, sx0, sy0, 0.0]
    popt, _ = curve_fit(gauss2d, (x, y), env.ravel(), p0=p0, maxfev=20000)
    return abs(popt[3]), abs(popt[4])


def gauss2d(coords, A, x0, y0, sx, sy, off):
    x, y = coords
    return (
        A * np.exp(-((x - x0) ** 2 / (2 * sx**2) + (y - y0) ** 2 / (2 * sy**2))) + off
    ).ravel()

import numpy as np
from scipy.optimize import curve_fit


def gauss2d(xy, A, x0, y0, sx, sy, off):
    x, y = xy
    return (off + A*np.exp(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))).ravel()

def moments(img):
    """Background-subtracted centroid + rms widths. Initial guess, never final."""
    ny, nx = img.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    off = np.median(img)
    w = np.clip(img - off, 0, None)
    tot = w.sum()
    if tot <= 0:
        raise RuntimeError("no positive signal")
    x0 = (w*x).sum()/tot
    y0 = (w*y).sum()/tot
    sx = max(np.sqrt((w*(x-x0)**2).sum()/tot), 1.0)
    sy = max(np.sqrt((w*(y-y0)**2).sum()/tot), 1.0)
    return np.array([w.max(), x0, y0, sx, sy, off])

def fit_centre(img, nsig=4, maxfev=5000):
    """
    Fit a 2D Gaussian. Returns [A, x0, y0, sx, sy, off] in this image's
    own pixel coords (x = column, y = row).
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    p0 = moments(img)

    # restrict to a box around the moment estimate so wings/fringes
    # don't drag the fit
    m = ((x - p0[1])**2/(nsig*p0[3])**2 +
         (y - p0[2])**2/(nsig*p0[4])**2) <= 1
    if m.sum() < 20:
        m = np.ones_like(img, bool)

    lo = [0,      0,  0,  0.5, 0.5, -np.inf]
    hi = [np.inf, nx, ny, nx,  ny,   np.inf]
    p, cov = curve_fit(gauss2d, (x[m], y[m]), img[m],
                       p0=p0, bounds=(lo, hi), maxfev=maxfev)
    return p, np.sqrt(np.diag(cov))

def crop_about(img, centre_rc, half):
    """Crop `img` to (2*hy+1, 2*hx+1) about integer centre (row, col)."""
    (cy, cx), (hy, hx) = centre_rc, half
    return img[cy-hy:cy+hy+1, cx-hx:cx+hx+1]

def max_half(centre_rc, shape):
    c = np.asarray(centre_rc)
    return np.minimum(c, np.asarray(shape) - 1 - c)

def align_clouds(imgA, imgB, half=None):
    pA, pB = fit_centre(imgA), fit_centre(imgB)      # A, x0, y0, sx, sy, off
    cA = np.array([pA[2], pA[1]])
    cB = np.array([pB[2], pB[1]])
    iA, iB = np.round(cA).astype(int), np.round(cB).astype(int)

    if half is None:
        half = np.minimum(max_half(iA, imgA.shape), max_half(iB, imgB.shape))
    return crop_about(imgA, iA, half), crop_about(imgB, iB, half), pA, pB

def align_stacks(setA, setB):
    fitsA = [fit_centre(im)[0] for im in setA]
    fitsB = [fit_centre(im)[0] for im in setB]
    cA = [np.round([p[2], p[1]]).astype(int) for p in fitsA]
    cB = [np.round([p[2], p[1]]).astype(int) for p in fitsB]

    half = np.minimum.reduce(
        [max_half(c, im.shape) for c, im in zip(cA, setA)] +
        [max_half(c, im.shape) for c, im in zip(cB, setB)]
    )
    A = np.stack([crop_about(im, c, half) for im, c in zip(setA, cA)])
    B = np.stack([crop_about(im, c, half) for im, c in zip(setB, cB)])
    return A, B, half, fitsA, fitsB
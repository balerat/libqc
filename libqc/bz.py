
def bz_mask(kx, ky, half_side, angle_deg):
    """Boolean mask: True inside an origin-centered square of given half-side, rotated."""
    a = np.deg2rad(angle_deg)
    kxp = kx * np.cos(a) + ky * np.sin(a)
    kyp = -kx * np.sin(a) + ky * np.cos(a)
    return (np.abs(kxp) <= half_side) & (np.abs(kyp) <= half_side)


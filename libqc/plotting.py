import numpy as np
from matplotlib.colors import LinearSegmentedColormap

colors = {
    "red":    {"base": "#E51E32", "light": "#ED6170", "dark": "#A01523"},
    "orange": {"base": "#FF782A", "light": "#FFA169", "dark": "#B2541D"},
    "amber":  {"base": "#FDA805", "light": "#FEC250", "dark": "#B17503"},
    "yellow": {"base": "#E2CF04", "light": "#EBDD50", "dark": "#9E9103"},
    "lime":   {"base": "#B1CA05", "light": "#C8D950", "dark": "#7C8D03"},
    "green":  {"base": "#98C217", "light": "#B7D45C", "dark": "#6A8710"},
    "olive":  {"base": "#779815", "light": "#A0B75B", "dark": "#536A0E"},
    "teal":   {"base": "#029E77", "light": "#4EBB9F", "dark": "#016E53"},
    "pine":   {"base": "#09989C", "light": "#53B7BA", "dark": "#066A6D"},
    "cyan":   {"base": "#059CCD", "light": "#50BAE0", "dark": "#036D8F"},
    "blue":   {"base": "#3F64CE", "light": "#7892DD", "dark": "#2C4690"},
    "purple": {"base": "#7E2B8E", "light": "#A56BAF", "dark": "#581E63"},
}

# transparent -> dark red, used for optical-density overlays
od_cmap = LinearSegmentedColormap.from_list(
    "od_cmap", [(0, (1.0, 0, 0, 0.0)), (1, (0.5, 0.0, 0.0, 1.0))]
)


def make_alpha_cmap(rgba, name: str) -> LinearSegmentedColormap:
    """Build a transparent-to-opaque colormap of a single colour.

    Useful to overlay several ``imshow`` layers (one colour per dataset)
    without the low values of one hiding the others.

    :param rgba: base colour as an (r, g, b, a) tuple in [0, 1]
    :param name: matplotlib name for the new colormap
    """
    r, g, b, _ = rgba
    return LinearSegmentedColormap.from_list(
        name, [(0, (r, g, b, 0.0)), (1, (r, g, b, 1.0))]
    )


def square_corners(half_side: float, angle_deg: float = 0.0, center=(0.0, 0.0)):
    """Closed (x, y) arrays tracing a square — e.g. a Brillouin-zone boundary.

    :param half_side: half side length of the square
    :param angle_deg: rotation of the square in degrees (45 for the diagonal BZ)
    :param center: centre of the square
    :returns: (x, y) arrays of 5 points, ready for ``ax.plot(*square_corners(...))``
    """
    c = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * half_side
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = c @ rot.T + np.array(center)
    return c[:, 0], c[:, 1]


def draw_bz(ax, half_side):
    """Overlay the aligned and 45-degree pseudo-Brillouin-zone squares."""
    ax.plot(*square_corners(half_side, angle_deg=0), c="k", lw=1.5, label="BZ")
    ax.plot(
        *square_corners(half_side, angle_deg=45),
        c="k",
        lw=1.5,
        ls="--",
        label="BZ (45 deg)",
    )

def draw_bz_full(ax, hs, x_min, x_max, y_min, y_max):
    """Draw all the lines of the tof images"""
    lines1 = []
    # First zone
    lines1.append(np.array([[-x_max, -hs/2], [x_max, -hs/2]]))
    lines1.append([[-x_max, hs/2], [x_max, hs/2]])
    lines1.append([[-hs/2, -y_max], [-hs/2, y_max]])
    lines1.append([[hs/2, -y_max], [hs/2, y_max]])
    for line in lines1:
        ax.plot(line[:0], line[:1], c="k", lw=1.5)





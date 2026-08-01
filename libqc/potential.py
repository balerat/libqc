import numpy as np

from libqc.transform import stretch_coord

def qc_potential(x, y, k_lattice, phi_x, phi_y, phi_t, phi_d, theta=0):
    """Eight-fold quasicrystal potential on a (optionally rotated) grid.

    :param x, y: coordinate meshgrids
    :param k_lattice: lattice wavevector in the inverse unit of x/y
    :param phi_x, phi_y, phi_t, phi_d: phase of each of the 4 lattice axes
    :param theta: rotation of the lattice relative to the grid [rad]
    :returns: dimensionless potential in [0, 4] (multiply by the depth)
    """

    x_rotated = x * np.cos(theta) - y * np.sin(theta)
    y_rotated = x * np.sin(theta) + y * np.cos(theta)

    V_x = np.sin(k_lattice * (x_rotated) + phi_x)**2
    V_y = np.sin(k_lattice * (y_rotated) + phi_y)**2
    V_t = np.sin(k_lattice * (x_rotated + y_rotated) / np.sqrt(2) + phi_t)**2
    V_d = np.sin(k_lattice * (x_rotated - y_rotated) / np.sqrt(2) + phi_d)**2

    return V_x + V_y + V_t + V_d


def qc_potential_light(x, y, k_lattice, phi_x, phi_y, phi_t, phi_d):
    """Quasicrystal potential without rotation — fast path for theory grids.

    Same as :func:`qc_potential` with ``theta=0``; kept as its own njit
    function so the rotation branch is compiled away.
    """
    V_x = np.sin(k_lattice * x + phi_x) ** 2
    V_y = np.sin(k_lattice * y + phi_y) ** 2
    V_t = np.sin(k_lattice * (x + y) / np.sqrt(2) + phi_t) ** 2
    V_d = np.sin(k_lattice * (x - y) / np.sqrt(2) + phi_d) ** 2
    return V_x + V_y + V_t + V_d


def qc_potential_general(x, y, params):
    '''
    Function to calculate a potential in rotated coordinate and with a shifted coordinate
    '''
    theta = params[5]
    k_lattice = params[0]
    phi_x, phi_y, phi_t, phi_d = params[1:5]
    x_t, y_t = stretch_coord(x, y, *params[6:])

    x_rotated = x_t * np.cos(theta) - y_t * np.sin(theta)
    y_rotated = x_t * np.sin(theta) + y_t * np.cos(theta)

    V_x = np.sin(k_lattice * (x_rotated) + phi_x)**2
    V_y = np.sin(k_lattice * (y_rotated) + phi_y)**2
    V_t = np.sin(k_lattice * (x_rotated + y_rotated) / np.sqrt(2) + phi_t)**2
    V_d = np.sin(k_lattice * (x_rotated - y_rotated) / np.sqrt(2) + phi_d)**2

    return V_x + V_y + V_t + V_d

def square_potential(x, y, k_lattice, phi_x, phi_y, theta=0):
    x_rotated = x * np.cos(theta) - y * np.sin(theta)
    y_rotated = x * np.sin(theta) + y * np.cos(theta)

    V_x = np.sin(k_lattice * (x_rotated) + phi_x)**2
    V_y = np.sin(k_lattice * (y_rotated) + phi_y)**2

    return V_x + V_y

def square_potential_general(x, y, params):
    theta = params[3]
    k_lattice = params[0]
    phi_x, phi_y = params[1:3]
    x_t, y_t = stretch_coord(x, y, *params[4:])

    x_rotated = x_t * np.cos(theta) - y_t * np.sin(theta)
    y_rotated = x_t * np.sin(theta) + y_t * np.cos(theta)


    V_x = np.sin(k_lattice * (x_rotated) + phi_x)**2
    V_y = np.sin(k_lattice * (y_rotated) + phi_y)**2

    return V_x + V_y


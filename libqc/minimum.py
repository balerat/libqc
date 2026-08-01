import numpy as np
from scipy.ndimage import minimum_filter

from libqc.misc import generate_xy_list
from libqc.potential import qc_potential, qc_potential_general, square_potential_general
from libqc.transform import rotate, stretch_coord
from libqc.octagon import get_octagon_pos

def find_qc_minimum(img, params, size):
    '''
    Find the minimum of the quasicrystal potential with ndimage.minimum_filter, get its coordinates on the octogon
    and verifies that it lies within its bounds.
    '''
    theta = params[5]
    k_lattice = params[0]
    phi_arr = params[1:5]
    x_list, y_list = generate_xy_list(img)

    potential = qc_potential(x_list, y_list, *params[:6])

    minimums = minimum_filter(potential, size=size, mode='nearest', cval=0.0) == potential
    x_min, y_min = x_list[np.where(minimums)], y_list[np.where(minimums)]
    index_x = np.where(minimums)[0]
    index_y = np.where(minimums)[1]

    rot_x_min, rot_y_min = rotate(x_min, y_min, theta)
    omega_x, omega_y = get_octagon_pos(rot_x_min, rot_y_min, k_lattice, *phi_arr)
    return x_min, y_min, omega_x, omega_y, [index_x, index_y]

def find_qc_minimum_general(img, params, size):
    '''
    Find the minimum of the quasicrystal potential with ndimage.minimum_filter, get its coordinates on the octogon
    and verifies that it lies within its bounds.
    '''
    theta = params[5]
    k_lattice = params[0]
    phi_arr = params[1:5]
    x_list, y_list = generate_xy_list(img)
    potential = qc_potential_general(x_list, y_list, params)

    minimums = minimum_filter(potential, size=size, mode='nearest', cval=0.0) == potential
    min_rows, min_cols = np.where(minimums)
    x_min, y_min = x_list[min_rows, min_cols], y_list[min_rows, min_cols]
    index_x = min_cols
    index_y = min_rows

    x_min_t, y_min_t = stretch_coord(x_min, y_min, *params[6:])
    rot_x_min, rot_y_min = rotate(x_min_t, y_min_t , theta)
    omega_x, omega_y = get_octagon_pos(rot_x_min, rot_y_min, k_lattice, *phi_arr)
    return x_min, y_min, omega_x, omega_y, [index_x, index_y]

def find_square_minimum_general(img, params, size):
    '''
    Find the minimum of the quasicrystal potential with ndimage.minimum_filter, get its coordinates on the octogon
    and verifies that it lies within its bounds.
    '''
    x_list, y_list = generate_xy_list(img)
    potential = square_potential_general(x_list, y_list, params)

    minimums = minimum_filter(potential, size=size, mode='nearest', cval=0.0) == potential
    x_min, y_min = x_list[np.where(minimums)], y_list[np.where(minimums)]
    index_x = np.where(minimums)[1]
    index_y = np.where(minimums)[0]

    return x_min, y_min, [index_x, index_y]


def find_qc_minimum_theo(x, y, params, size):
    """Locate quasicrystal minima on a *theory* grid (physical coordinates).

    Unlike :func:`find_qc_minimum`, which derives a pixel grid from an image,
    this works directly on the provided coordinate meshgrids — used by the
    split-step simulations to seed Wannier functions at lattice sites.

    :param x, y: coordinate meshgrids [m] (or any unit matching k_lattice)
    :param params: [k_lattice, phi_x, phi_y, phi_t, phi_d, theta]
    :param size: neighbourhood size (grid points) for the minimum filter
    :returns: ``(minimums, oct_mins, index_mins)`` — each of shape (2, N):
        real-space (x, y), octagon coordinates, and (row, col) grid indices.
    """
    k_lattice = params[0]
    phi_arr = params[1:5]
    theta = params[5]

    potential = qc_potential(x, y, *params[:6])
    is_min = minimum_filter(potential, size=size, mode="nearest", cval=0.0) == potential
    rows, cols = np.where(is_min)
    x_min, y_min = x[rows, cols], y[rows, cols]

    rot_x, rot_y = rotate(x_min, y_min, theta)
    omega_x, omega_y = get_octagon_pos(rot_x, rot_y, k_lattice, *phi_arr)

    minimums = np.stack([x_min, y_min])
    oct_mins = np.stack([omega_x, omega_y])
    index_mins = np.stack([rows, cols])
    return minimums, oct_mins, index_mins


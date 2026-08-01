import numpy as np

def get_octagon_pos(x, y, k_lattice, phi_x, phi_y, phi_t, phi_d):
    '''
    Transform real space minmum coordinates to octogon coordinates.
    '''
    theta_1 = k_lattice * x + phi_x
    theta_2 = k_lattice * y + phi_y
    theta_3 = k_lattice * (x + y) / np.sqrt(2) + phi_t
    theta_4 = k_lattice * (x - y) / np.sqrt(2) + phi_d
    
    theta_1 = theta_1 - np.pi * np.rint(theta_1 / np.pi)
    theta_2 = theta_2 - np.pi * np.rint(theta_2 / np.pi)
    theta_3 = theta_3 - np.pi * np.rint(theta_3 / np.pi)
    theta_4 = theta_4 - np.pi * np.rint(theta_4 / np.pi)
    
    theta_x = theta_1 - (theta_3 + theta_4) / np.sqrt(2)
    theta_y = theta_2 - (theta_3 - theta_4) / np.sqrt(2)
    
    return theta_x, theta_y



"""Physical and experimental constants for the quasicrystal experiment.

Everything is SI unless the name says otherwise. Species-independent
constants only — atom data lives in :mod:`libmbqd.species`, and
simulation-parameter bundles (g_2d, recoil units, ...) in
:class:`libmbqd.ssfm.GPEParams`.

Note there are two distinct camera pixel pitches:

* ``PIXELPITCH_QGM`` — quantum-gas-magnifier imaging camera (13/6 um),
  demagnified by the x50 magnifier into ``QGMPITCH``.
* ``PIXELPITCH_TOF`` — time-of-flight imaging camera (13/2 um), combined
  with ``TOF`` and ``TOF_MAG`` into the momentum calibration ``DK_TOF``.
"""

import numpy as np
from scipy.constants import atomic_mass, hbar, pi

PI = pi
HBAR = hbar



# --- lattice ---
WAVELENGTH = 726e-9                       # lattice beam wavelength [m]
K_LATTICE = 2 * pi / WAVELENGTH           # lattice wavevector [rad/m]
K0 = K_LATTICE                            # alias kept for older scripts

# --- K39 derived quantities ---
MK39 = 39 * atomic_mass               # [kg]
ER = (HBAR**2 * K_LATTICE**2) / (2 * MK39)  # recoil energy [J]
T_ER = 1 / hbar * ER 

A0 = 5.29e-11                             # Bohr radius [m]
G = (4 * pi * HBAR**2) / MK39    # 3D interaction strength [J m^3 a0-1]

# 2D reduction assuming a z-lattice of depth 20 Er
Z0 = np.sqrt(HBAR / MK39 / np.sqrt(2 * 20 * ER * K_LATTICE**2 / MK39))
G_2D = G / np.sqrt(2 * pi) / Z0           # 2D interaction strength [J m^2 a0-1]

# --- imaging ---
IMG_LAMBDA = 767e-9                       # imaging wavelength [m]
SIGMA0 = 3 * IMG_LAMBDA**2 / (2 * pi)     # resonant absorption cross-section [m^2]

# quantum-gas-magnifier camera
PIXELPITCH_QGM = 13 / 6 * 1e-6            # camera pixel pitch [m]
QGMPITCH = PIXELPITCH_QGM / 50            # effective pitch through the x50 magnifier [m]

# time-of-flight camera
PIXELPITCH_TOF = 13 / 2 * 1e-6            # camera pixel pitch [m]
TOF = 9e-3                                # time of flight [s]
TOF_MAG = 3                               # imaging magnification
DX_TOF = PIXELPITCH_TOF / TOF_MAG         # object-plane pixel size [m]
DK_TOF = MK39 * DX_TOF / (HBAR * TOF)  # momentum per pixel after TOF [rad/m]
DK_TOF = 155706.7520069962

# backwards-compatible aliases (older scripts used one name for both cameras)
PIXELPITCH = PIXELPITCH_QGM
QMGPITCH = QGMPITCH

WX_TRAP = 2 * pi * (41**2 + 909**2 / 4.9 * 0.15) ** 0.5
WY_TRAP = WX_TRAP

K0 = 2 * np.pi / WAVELENGTH

import copy
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm

from libqc.base.shot import Shot

log_info = "INFO: "


def create_cache_parser():
    """Deprecated — use :func:`libmbqd.cache.reload_requested` instead."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


# --- split-step results --- #

def load_ssfm(theo_path_file):
    """Load one ``out_imgt.npy`` into a DataFrame of energy traces.

    :param theo_path_file: path to the ``.npy`` file
    :returns: DataFrame with columns inter/potential/kinetic, indexed by
        time in microseconds
    """
    import pandas as pd

    print(log_info + "Loading SSFM")
    with open(theo_path_file, "rb") as f:
        data = np.load(f, allow_pickle=True).item()
    time_index = np.array(data["times"]) * 1e6

    return pd.DataFrame(
        {
            "inter": data["inter"],
            "potential": data["potential"],
            "kinetic": data["kinetic"],
        },
        index=time_index,
    )


def load_ssfm_dir(ssfm_path):
    """Load a directory of ``out_imgt.npy`` runs into per-quantity DataFrames.

    :param ssfm_path: directory whose children are the ``.npy`` files
    :returns: (interaction, potential, kinetic) DataFrames — one column per
        run, indexed by time in microseconds
    """
    import pandas as pd

    print(log_info + "Loading SSFM")
    inters, potentials, kinetics = [], [], []
    data = None
    for child in sorted(Path(ssfm_path).iterdir()):
        with open(child, "rb") as f:
            data = np.load(f, allow_pickle=True).item()
        inters.append(data["inter"])
        potentials.append(data["potential"])
        kinetics.append(data["kinetic"])
    if data is None:
        raise FileNotFoundError(f"No runs found in {ssfm_path}")
    time_index = np.array(data["times"]) * 1e6
    return (
        pd.DataFrame(np.array(inters).T, index=time_index),
        pd.DataFrame(np.array(potentials).T, index=time_index),
        pd.DataFrame(np.array(kinetics).T, index=time_index),
    )


# --- Hubbard parameters --- #

def load_uiiii_param(hubbard_param_path, lattice_depth=(3.0,)):
    """Load on-site interaction U_iiii and octagon coordinates per depth.

    :param hubbard_param_path: root containing ``results_depth_<d>`` folders
    :param lattice_depth: iterable of depths to load
    :returns: (u_data, oct_data) lists, one entry per depth
    """
    print(log_info + "Loading Uiiii")
    u_data, oct_data = [], []
    for depth in lattice_depth:
        folder = Path(hubbard_param_path) / f"results_depth_{depth:.4f}"
        u_data.append(np.load(folder / "U_iiii.npy"))
        oct_data.append(np.load(folder / "octagon.npy").copy())
    return u_data, oct_data


def load_hamiltonian(hubbard_param_path, lattice_depth):
    """Load sparse tight-binding Hamiltonians and octagon coords per depth.

    :param lattice_depth: iterable of depths to load
    :returns: (ham_data, oct_data) lists; Hamiltonians as ``scipy.sparse.csc_matrix``
    """
    print(log_info + "Loading Hamiltonian")
    ham_data, oct_data = [], []
    for depth in lattice_depth:
        folder = Path(hubbard_param_path) / f"results_depth_{depth:.4f}"
        ham_file = np.load(folder / "hamiltonian_real.npz")
        oct_data.append(np.load(folder / "octagon.npy").copy())
        ham_data.append(
            sparse.csc_matrix(
                (ham_file["data"], ham_file["indices"], ham_file["indptr"]),
                shape=ham_file["shape"],
            )
        )
    return ham_data, oct_data


# --- Wannier functions --- #

def load_theo_wannier(wannier_path: Path, lattice_depth: float) -> dict:
    """Load the integrated theory Wannier templates for one lattice depth.

    :returns: dict with (at least) keys ``"octagon"`` and ``"wannier"`` —
        feed to :func:`libmbqd.qgm.wannier.creating_interpolation`
    """
    file = Path(wannier_path) / f"integrate_wannier_out_{lattice_depth}.npy"
    with open(file, "rb") as f:
        data = np.load(f, allow_pickle=True).item()
    print(log_info + f"Loaded wannier keys: {list(data.keys())}")
    return data


def load_fitted_wannier(wannier_path, n_row, n_col):
    """Load a fitted-Wannier pipeline output (``out.pkl``) into grids.

    :param n_row, n_col: shape of the scan grid the shots were taken on
    :returns: (wannier_img, population, omega_x, omega_y) arrays of shape
        (n_row, n_col); the last three have object dtype (ragged per shot)
    """
    print(log_info + "Loading wannier functions")
    with open(Path(wannier_path) / "out.pkl", "rb") as f:
        data = pickle.load(f)

    n = len(data["wannier_list"])
    wannier_img = data["wannier_img"].reshape(n_row, n_col)
    population = np.empty(n, dtype=object)
    omega_x = np.empty(n, dtype=object)
    omega_y = np.empty(n, dtype=object)
    for i in range(n):
        population[i] = np.array([w.sum() for w in data["wannier_list"][i]])
        omega_x[i] = data["omega_x_dens"][i]
        omega_y[i] = data["omega_y_dens"][i]

    return (
        wannier_img,
        population.reshape(n_row, n_col),
        omega_x.reshape(n_row, n_col),
        omega_y.reshape(n_row, n_col),
    )


def load_phases(phase_path, n_row, n_col):
    """Load fitted lattice phases (``out.pkl``) into an (n_row, n_col, 11) array."""
    print(log_info + "Loading phases")
    with open(Path(phase_path) / "out.pkl", "rb") as f:
        data = pickle.load(f)
    fitting_params = np.zeros((n_row, n_col, 11))
    for i in range(n_row):
        for j in range(n_col):
            fitting_params[i, j] = np.array(data["fitting_parameter"][n_col * i + j])
    return fitting_params


# --- experimental shots --- #

def list_shot_files(folders) -> list[tuple[str, str]]:
    """All ``.hdf5`` shots in ``folders`` (excluding ``*analysis.hdf5``).

    :returns: sorted (folder, filename) pairs, folder order preserved
    """
    pairs = []
    for folder in folders:
        names = sorted(
            f.name
            for f in Path(folder).iterdir()
            if f.is_file() and f.suffix == ".hdf5" and not f.name.endswith("analysis.hdf5")
        )
        pairs.extend((str(folder), name) for name in names)
    return pairs


def _clean(img: np.ndarray) -> np.ndarray:
    """Replace inf/nan by the max finite value and flip vertically."""
    bad = ~np.isfinite(img)
    if bad.any():
        img[bad] = img[~bad].max()
    return np.flipud(img)


def load_shots(
    folders,
    variables,
    rois: dict,
    *,
    frametype: str = "OD",
    od_recalculate: bool = False,
    n_max: int | None = None,
    files: list[tuple[str, str]] | None = None,
    verbose: bool = False,
):
    """Load HDF5 shots into per-ROI image stacks plus run parameters.

    The ``"bg"`` ROI is mandatory: its mean is subtracted from the ``"main"``
    images (background correction). Shots that fail to load are skipped with
    a warning.

    :param folders: folder or list of folders containing the ``.hdf5`` shots
    :param variables: userdata keys to collect per shot ("timestamp" allowed)
    :param rois: dict of name -> (row0, row1, col0, col1); must contain
        ``"main"`` and ``"bg"``
    :param frametype: frame to analyse (default optical density "OD")
    :param od_recalculate: recompute the OD instead of using the stored one
    :param n_max: stop after this many shots (default: all)
    :param files: explicit (folder, filename) pairs, bypassing the directory scan
    :returns: ``(images, params)`` — ``images[name]`` is an (n, h, w) stack
        for every non-bg ROI, ``params[var]`` an array of length n
    """
    if isinstance(folders, (str, Path)):
        folders = [folders]
    if "main" not in rois or "bg" not in rois:
        raise ValueError("rois must contain 'main' and 'bg'")

    if files is None:
        files = list_shot_files(folders)
    if n_max is not None:
        files = files[:n_max]

    image_rois = {name: roi for name, roi in rois.items() if name != "bg"}
    images = {
        name: np.zeros((len(files), roi[1] - roi[0], roi[3] - roi[2]))
        for name, roi in image_rois.items()
    }
    records: list[dict] = []
    kept = 0

    print(log_info + f"Loading {len(files)} shots")
    for folder, filename in tqdm(files):
        try:
            run = Shot(folder, filename)
            record = {
                v: (run.timestamp if v == "timestamp" else run.userdata[v])
                for v in variables
            }
            frame = run.get_frame(frametype=frametype, od_recalculate=od_recalculate)

            bg = copy.copy(frame)
            bg.roi = rois["bg"]
            bg_mean = np.mean(bg.image_cropped)

            for name, roi in image_rois.items():
                view = copy.copy(frame)
                view.roi = roi
                img = view.image_cropped
                if name == "main":
                    img = img - bg_mean
                images[name][kept] = _clean(img)
        except Exception as exc:  # damaged shot: skip, keep stack aligned
            print(log_info + f"skipping {filename}: {exc}")
            continue
        records.append(record)
        kept += 1

    images = {name: stack[:kept] for name, stack in images.items()}
    params = {
        v: np.array([r[v] for r in records]) for v in (records[0] if records else {})
    }
    if verbose:
        print(log_info + f"loaded {kept}/{len(files)} shots")
    return images, params


def loadData(
    folders, variables, rois, file_in=None, NRow=0, NCol=0,
    analysisframetype="OD", od_recalculate=False, vmin=0, vmax=2,
    my_dir_save=None, plotImgs=True, verboseLog=False,
    titleVals=None, titleValUnits=None, verbose=False,
):
    """Old interface, kept for existing scripts — use :func:`load_shots`.

    Returns ``(ImgArr, r)`` or ``(ImgArr, ExtraImgArrs, r)`` when extra ROIs
    are given, exactly as before. The plotting-related arguments are accepted
    and ignored (they never had an effect).
    """
    n_max = NRow * NCol if NRow * NCol else None
    files = [(str(folders[0]), str(file_in))] if file_in is not None else None
    images, params = load_shots(
        folders, variables, rois,
        frametype=analysisframetype, od_recalculate=od_recalculate,
        n_max=n_max, files=files, verbose=verbose,
    )
    extra = {name: stack for name, stack in images.items() if name != "main"}
    if extra:
        return images["main"], extra, params
    return images["main"], params

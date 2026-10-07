# --- experimental shots --- #
import copy
import pickle
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from tqdm import tqdm

from libqc.base.shot import Shot

N_JOBS = 8  # parallel worker processes for shot loading (1 = serial)
log_info = "INFO: "

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


def _load_one(folder, filename, variables, rois, image_rois, frametype, od_recalculate):
    """Load one shot. Returns (record, images) or (None, error message)."""
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

        imgs = {}
        for name, roi in image_rois.items():
            view = copy.copy(frame)
            view.roi = roi
            img = np.array(view.image_cropped, dtype=float)
            if name == "main":
                img = img - bg_mean
            imgs[name] = _clean(img)
        return record, imgs
    except Exception as exc:  # damaged shot: skip, keep stack aligned
        return None, f"skipping {filename}: {exc}"


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
    a warning. Shots are loaded in parallel (``N_JOBS`` processes); the
    output order is the file order.

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

    print(log_info + f"Loading {len(files)} shots")
    out = Parallel(n_jobs=N_JOBS)(
        delayed(_load_one)(
            folder, filename, variables, rois, image_rois, frametype, od_recalculate
        )
        for folder, filename in tqdm(files)
    )

    records: list[dict] = []
    stacks = {name: [] for name in image_rois}
    for record, payload in out:
        if record is None:
            print(log_info + payload)
            continue
        records.append(record)
        for name in image_rois:
            stacks[name].append(payload[name])
    kept = len(records)

    images = {
        name: (
            np.stack(stacks[name])
            if kept
            else np.zeros((0, roi[1] - roi[0], roi[3] - roi[2]))
        )
        for name, roi in image_rois.items()
    }
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

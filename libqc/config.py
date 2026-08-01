"""Per-machine paths and compute device.

Single place that answers "where is the data and what hardware am I on?",
so the same script runs unmodified on the MacBook (mps, /Volumes mount),
the cluster (cuda, gvfs SMB mount) or a Windows machine.

Exposed constants
-----------------
IFS_PATH      root of the Quasicrystal tree on the IFS share
RESULTS_PATH  personal results area on the share (IFS_PATH / CRSID)
PHD_PATH      local root of the phd working tree
TORCH_DEVICE  default torch device name for this platform ("cuda"/"mps"/"cpu")

Use :func:`get_device` rather than TORCH_DEVICE directly — it honours the
``MBQD_DEVICE`` environment variable and degrades to cpu when the
accelerator is unavailable.
"""

import os
import platform
import subprocess
from pathlib import Path

CRSID = "bl555"

system = platform.system()

if system == "Linux":
    _uid = subprocess.check_output(["id", "-u"], text=True).strip()
    IFS_PATH = Path(
        f"/run/user/{_uid}/gvfs/smb-share:server=ifs-prod-943-cifs.ifs.uis.private.cam.ac.uk,"
        "share=mbqd_data/Quasicrystal"
    )
    PHD_PATH = Path.home() / "phd"
    TORCH_DEVICE = "cuda"

elif system == "Darwin":
    IFS_PATH = Path("/Volumes/MBQD_data/Quasicrystal")
    PHD_PATH = Path.home() / "Code/phd"
    TORCH_DEVICE = "mps"

elif system == "Windows":
    IFS_PATH = Path(r"\\ifs-prod-943-cifs.ifs.uis.private.cam.ac.uk\MBQD_data\Quasicrystal")
    PHD_PATH = Path.home() / "Code/phd"
    TORCH_DEVICE = "cpu"

else:
    raise RuntimeError(f"[device] Unknown platform: {system}")

RESULTS_PATH = IFS_PATH / CRSID


def require(path: Path) -> Path:
    """Return ``path`` if it exists, else raise with a hint about the mount.

    Use on IFS paths before long jobs so a missing SMB mount fails loudly
    at startup instead of at save time.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — is the IFS share mounted on this machine?"
        )
    return path


def get_device(override: str | None = None):
    """Return the torch device to compute on.

    Priority: ``override`` argument > ``MBQD_DEVICE`` env var > platform
    default (cuda on Linux, mps on macOS), falling back to cpu when the
    accelerator is not actually available.
    """
    import torch  # local import: keep libmbqd importable without torch

    name = override or os.environ.get("MBQD_DEVICE") or TORCH_DEVICE
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    return torch.device(name)

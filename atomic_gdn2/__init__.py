"""atomic_gdn2 — Gated DeltaNet-2 with Block-Local Rescaling Pallas kernels (TPU v5e)."""
from importlib.metadata import version as _version, PackageNotFoundError as _PkgNotFound

from .config import BLRConfig, make_cfg, PROD
from .pipeline import make_blr_trainable

try:
    __version__ = _version("atomic-ops")
except _PkgNotFound:
    __version__ = "0.2.0"

"""Atlas-In-Diffusion: recover a population brain atlas from a pretrained diffusion
model by running its reverse process deterministically. See recover.py."""
import os as _os, sys as _sys

# make the bundled model code (ddpm/, AutoEncoder/) importable everywhere
_VENDOR = _os.path.join(_os.path.dirname(__file__), "_vendor")
if _VENDOR not in _sys.path:
    _sys.path.insert(0, _VENDOR)

"""Checkpoint resolution.

  age_cond.pt            age-conditioned generator (this work) -- auto from our HF repo
  PatchVolume4x_s2.ckpt  frozen 4x VQ autoencoder  -- from 3D-MedDiffusion (Google Drive)
  BiFlowNet_4x.pt        original generator        -- from 3D-MedDiffusion (Google Drive)

We do NOT re-host 3D-MedDiffusion's weights. Download them once from the authors'
Google Drive and drop them in ATLAS_WEIGHTS_DIR (default ~/.cache/atlas_in_diffusion):

    https://drive.google.com/drive/folders/1h1Ina5iUkjfSAyvM5rUs4n1iqg33zB-J
"""
import os
from huggingface_hub import hf_hub_download

HF_REPO = "shijianjian/Atlas-In-Diffusion"          # our age model + example atlases
DRIVE_URL = "https://drive.google.com/drive/folders/1h1Ina5iUkjfSAyvM5rUs4n1iqg33zB-J"
WEIGHTS_DIR = os.environ.get(
    "ATLAS_WEIGHTS_DIR", os.path.expanduser("~/.cache/atlas_in_diffusion"))


def _local(fname):
    """Resolve a 3D-MedDiffusion checkpoint from ATLAS_WEIGHTS_DIR, else explain."""
    path = os.path.join(WEIGHTS_DIR, fname)
    if os.path.exists(path):
        return path
    raise SystemExit(
        f"\nMissing 3D-MedDiffusion checkpoint: {fname}\n"
        f"Download it from 3D-MedDiffusion's release (Google Drive):\n"
        f"    {DRIVE_URL}\n"
        f"and place it in {WEIGHTS_DIR}/  (or set ATLAS_WEIGHTS_DIR to its folder).\n")


def age_generator():
    """Age-conditioned brain generator (this work)."""
    return hf_hub_download(HF_REPO, "age_cond.pt")


def autoencoder():
    """Frozen 4x VQ autoencoder used to decode latents (3D-MedDiffusion)."""
    return _local("PatchVolume4x_s2.ckpt")


def base_generator():
    """Original pretrained generator (3D-MedDiffusion)."""
    return _local("BiFlowNet_4x.pt")


# 2D pixel-space DDPMs trained from scratch per domain (this work). Filenames on
# the HF repo; a copy in ATLAS_WEIGHTS_DIR overrides the download.
DOMAIN_2D_FILES = {
    "chestxray": "ddpm2d_chestxray_128.pt",
}


def domain_2d(name):
    """Fetch a 2D domain checkpoint (this work), from ATLAS_WEIGHTS_DIR or HF."""
    fname = DOMAIN_2D_FILES[name]
    local = os.path.join(WEIGHTS_DIR, fname)
    if os.path.exists(local):
        return local
    return hf_hub_download(HF_REPO, fname)

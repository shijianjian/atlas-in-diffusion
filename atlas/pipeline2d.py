"""Deterministic atlas recovery for the 2D pixel-space DDPMs.

Same idea as the 3D path, but the generator is a small UNet DDPM trained from
scratch per domain (chest X-ray, ...), operating directly in image space — no
autoencoder. Recovery drops the noise term and marches the reverse process to t*.
"""
import os
import torch
from torchvision.utils import save_image

from . import hf, pipeline           # reuse enable_determinism + DEVICE
from ddpm2d.trajectory import Trajectory
from ddpm2d.model import Unet

DEVICE = pipeline.DEVICE
IMAGE_SIZE = 128                     # released 2D models are 128x128
CHANNELS = 3
T_START = 990                       # reverse process starts here (matches the paper's 2D recovery)
TSTAR_2D = 0                        # 2D models march to t=1 by default (pixel space, no early stop)

DOMAINS = ("chestxray",)            # cherry-picked demo domains


def load_model(domain):
    """Build the 2D UNet DDPM and load the domain checkpoint (downloads if needed)."""
    if domain not in DOMAINS:
        raise SystemExit(f"unknown 2D domain '{domain}'; available: {', '.join(DOMAINS)}")
    unet = Unet(dim=64, dim_mults=(1, 2, 4, 8), dilation=1, channels=CHANNELS).to(DEVICE)
    diff = Trajectory(unet, image_size=IMAGE_SIZE, timesteps=1000,
                      sampling_timesteps=250, loss_type="l1", local_randn=False).to(DEVICE)
    sd = torch.load(hf.domain_2d(domain), map_location="cpu")
    diff.load_state_dict(sd["model"])           # matches the training/eval checkpoint layout
    diff.eval()
    return diff


def recover(diff, *, tstar=TSTAR_2D, t_start=T_START, seed=None):
    """Deterministic recovery: x_T ~ N(0,I) -> compute_atlas -> image in [-1, 1]."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    timesteps = list(range(t_start, tstar, -1))
    with torch.no_grad():
        x = torch.randn(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        atlas = diff.compute_atlas(x, timesteps)
    return atlas.detach().cpu()


def save(image, stem):
    """Write <stem>.png (2D atlases are single images, not volumes)."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    save_image((image + 1) / 2, stem + ".png")
    return stem + ".png"

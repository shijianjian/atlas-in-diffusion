"""Deterministic atlas recovery.

The atlas is the image the reverse process converges to when the stochastic noise
term is dropped: iterate x_{t-1} = mu_theta(x_t, t) from x_T ~ N(0, I) to t*, then
decode through the frozen autoencoder. No registration, fine-tuning, or objective.
"""
import os
import numpy as np
import torch
import torchio as tio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import hf                       # noqa: F401 (also triggers _vendor path setup via package __init__)
from ddpm.BiFlowNet import BiFlowNet
from ddpm.trajectory import Trajectory
from AutoEncoder.model.PatchVolume import patchvolumeAE

# --- constants that must match the released checkpoints -----------------------
BIFLOWNET_KWARGS = dict(
    dim=72, dim_mults=[1, 1, 2, 4, 8], channels=8, init_kernel_size=3,
    cond_classes=7, learn_sigma=False, use_sparse_linear_attn=[0, 0, 0, 1, 1],
    vq_size=64, num_mid_DiT=1, patch_size=2,
)
CLASS_BRAIN_T1 = 3
LATENT_SHAPE = (8, 48, 48, 48)
RES_TOKEN = (48.0, 48.0, 48.0)
TIMESTEPS = 1000
TSTAR = 99
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def enable_determinism(seed=0):
    """Make recovery bit-reproducible (the model's 3D convs are otherwise
    nondeterministic under cuDNN). Set env ATLAS_NONDETERMINISTIC=1 to opt out."""
    if os.environ.get("ATLAS_NONDETERMINISTIC") == "1":
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_models(age_conditioned):
    """Return (generator, diffusion, autoencoder), downloading weights if needed."""
    ckpt = hf.age_generator() if age_conditioned else hf.base_generator()
    model = BiFlowNet(**BIFLOWNET_KWARGS, age_condition=age_conditioned).to(DEVICE)
    sd = torch.load(ckpt, map_location="cpu"); sd = sd.get("ema", sd)
    model.load_state_dict(sd, strict=False)
    model.train()                                  # pretrained model recovers in train() mode
    diff = Trajectory(channels=8, timesteps=TIMESTEPS, loss_type="l1").to(DEVICE)
    ae = patchvolumeAE.load_from_checkpoint(hf.autoencoder()).to(DEVICE).eval()
    return model, diff, ae


def _decode(ae, latent):
    emn, emx = ae.codebook.embeddings.min(), ae.codebook.embeddings.max()
    sc = (((latent + 1.0) / 2.0) * (emx - emn)) + emn
    v = ae.decode(sc, quantize=True).detach().squeeze(0).cpu().float()
    return v.transpose(1, 3).transpose(1, 2)       # (1, D, H, W)


def recover(model, diff, ae, *, age=None, cls=CLASS_BRAIN_T1,
            tstar=TSTAR, seed=0, cfg_scale=1.0):
    """Recover one atlas volume. age in years (or None for the base model)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    y = torch.tensor([cls], device=DEVICE)
    res = (torch.tensor(RES_TOKEN) / 64.0).unsqueeze(0).to(DEVICE)
    age_t = None if age is None else torch.tensor([age / 100.0], device=DEVICE)
    ts = list(range(TIMESTEPS - 1, tstar, -1))

    def denoise(x, t, y=None, res=None, **kw):
        if age_t is None:
            return model(x, t, y=y, res=res, age=None)
        a = age_t.expand(x.shape[0])
        cond = model(x, t, y=y, res=res, age=a)
        if cfg_scale == 1.0:
            return cond
        null = model(x, t, y=y, res=res, age=torch.full_like(a, -1.0))
        return null + cfg_scale * (cond - null)

    with torch.no_grad():
        z = torch.randn(1, *LATENT_SHAPE, device=DEVICE)
        latent, _, _ = diff.compute_deformation_field(denoise, z, y=y, res=res, timesteps=ts)
        return _decode(ae, latent)


def save(volume, stem):
    """Write <stem>.nii.gz and a <stem>.png mid-sagittal preview."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    tio.ScalarImage(tensor=volume.reshape(1, *volume.shape[-3:])).save(stem + ".nii.gz")
    mid = volume.reshape(*volume.shape[-3:]).numpy()
    mid = mid[mid.shape[0] // 2]
    lo, hi = np.percentile(mid, (1, 99))
    plt.imsave(stem + ".png", np.rot90(np.clip((mid - lo) / (hi - lo + 1e-8), 0, 1)), cmap="gray")
    return stem + ".nii.gz"

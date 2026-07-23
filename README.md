# Atlas-In-Diffusion

**A diffusion model trained only to synthesize images already contains the atlas of
its training population — recover it in one command.** Run the model's reverse
process *deterministically* (drop the noise term) and independent noise seeds
converge to the same image: the population template. No registration, no
fine-tuning, no atlas-specific objective.

From *"Atlases Are Already Inside: Recovering Population Templates from Pretrained
Diffusion Models."*

<table align="center">
  <tr>
    <td align="center"><img src="examples/anatomies/MRT1Brain.png" width="180"/></td>
    <td align="center"><img src="examples/anatomies/MRT2Brain.png" width="180"/></td>
    <td align="center"><img src="examples/anatomies/CTLegs.png" width="180"/></td>
  </tr>
  <tr>
    <td align="center">T1 brain</td>
    <td align="center">T2 brain</td>
    <td align="center">leg CT</td>
  </tr>
</table>

## Quick start

```bash
pip install -r requirements.txt      # (install torch for your CUDA build first — see requirements.txt)
python recover.py                    # T1 brain atlas -> outputs/atlas.nii.gz (+ preview PNG)
```

This recovers the T1 brain atlas from the original pretrained generator — the
paper's headline result. It needs the 3D-MedDiffusion weights (one-time download,
see below).

### Which model

The generator is selected by the flags — no separate model switch:

| command | model used |
|---|---|
| `python recover.py` | **base** generator (default) |
| `python recover.py --class 4` | **base** generator, a different anatomy class |
| `python recover.py --age 70` | **age-conditioned** generator (experimental, see below) |

Passing `--age`/`--ages` loads the age model; anything else uses the base model.

## Weights

- **3D-MedDiffusion weights** — download from the authors' Google Drive (we do not
  re-host them):
  **https://drive.google.com/drive/folders/1h1Ina5iUkjfSAyvM5rUs4n1iqg33zB-J**
  and place them in `~/.cache/atlas_in_diffusion/` (or any folder, then set
  `ATLAS_WEIGHTS_DIR`):
  - `PatchVolume4x_s2.ckpt` — the autoencoder (required for every atlas)
  - `BiFlowNet_4x.pt` — the base generator (for the default / `--class` atlases)
- **Age model** (`age_cond.pt`) — fetched automatically from our
  [HF repo](https://huggingface.co/shijianjian/Atlas-In-Diffusion) when you use `--age`.

## More (base model)

```bash
python recover.py                          # T1 brain atlas
python recover.py --class 4                # a different anatomy class
```

| flag | meaning | default |
|------|---------|---------|
| `--class` | anatomy-class token for the base model (3 = T1 brain) | 3 |
| `--out` | output directory | `outputs/` |
| `--seed` | optional seed for a fixed initial `x_T` | unset |
| `--tstar` | early-stopping time (lower = more detail) | 99 |

## Age-conditioned atlases

```bash
python recover.py --age 70                 # one age
python recover.py --ages 20,40,60,80       # a family
```

> **Note.** The age-conditioned generator is a proof-of-concept fine-tuned on a
> **small dataset**. Its atlases are softer, and — unlike the base model — recovery
> is **not fully stable across random initializations**. Treat these results as
> illustrative of the *conditioning* idea, not as production atlases. The paper's
> quantitative age results average several random-seed recoveries to smooth out
> this variation.

Age flags: `--age N` / `--ages a,b,c` (years), `--cfg` (age guidance scale, default 1.0).

## How it works

`recover.py` draws `x_T ~ N(0, I)` in the model's latent space, fixes the anatomy
(and optional age) conditioning, iterates the posterior-mean update
`x_{t-1} = μ_θ(x_t, t)` (no noise term) down to `t*`, and decodes through the frozen
autoencoder. Pass `--seed` for bit-reproducible output (it pins the RNG and selects
deterministic cuDNN kernels); without it, runs vary slightly at the sub-voxel level.

## Requirements

A CUDA GPU with **≥ 40 GB** memory is recommended for the 3D model. Runs on CPU but
very slowly. Tested with Python 3.11, PyTorch 2.1.2 (CUDA 11.8).

## Example atlases

`examples/` holds PNG previews; the full NIfTI volumes are on
[Hugging Face](https://huggingface.co/shijianjian/Atlas-In-Diffusion).


## Bundled model code
`atlas/_vendor/ddpm/` and `atlas/_vendor/AutoEncoder/` are a frozen copy of the
model code from **3D-MedDiffusion**
(https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion, arXiv:2412.13059), with
age-conditioning added to `BiFlowNet.py` and the deterministic recovery operator in
`trajectory.py` contributed by this work. Respect the 3D-MedDiffusion license and
retain this attribution. 3D-MedDiffusion itself builds on
[latent-diffusion](https://github.com/CompVis/latent-diffusion) and
[medicaldiffusion](https://github.com/firasgit/medicaldiffusion).

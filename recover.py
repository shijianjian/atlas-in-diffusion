#!/usr/bin/env python3
"""Recover a population brain atlas from a pretrained diffusion model.

Quick start (weights download automatically on first run):

    python recover.py                      # age-50 brain atlas  -> outputs/age50.nii.gz
    python recover.py --age 70             # a different age
    python recover.py --ages 20,40,60,80   # a whole age family
    python recover.py --base               # original model's brain atlas (no age)

Outputs a NIfTI volume plus a PNG preview for each atlas.
"""
import argparse
from atlas import pipeline as P


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--age", type=float, help="single age in years (age-conditioned model)")
    g.add_argument("--ages", help="comma-separated ages, e.g. 20,40,60")
    g.add_argument("--base", action="store_true",
                   help="use the original (non-age) generator")
    ap.add_argument("--class", dest="cls", type=int, default=P.CLASS_BRAIN_T1,
                    help=f"anatomy-class token for --base (default {P.CLASS_BRAIN_T1}=brain-T1)")
    ap.add_argument("--out", default="outputs", help="output directory (default: outputs/)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tstar", type=int, default=P.TSTAR, help="early-stopping time")
    ap.add_argument("--cfg", type=float, default=1.0, help="age guidance scale (>1)")
    args = ap.parse_args()

    if args.base:
        ages = [None]
    elif args.ages:
        ages = [float(a) for a in args.ages.split(",")]
    else:
        ages = [args.age if args.age is not None else 50.0]

    print(f"device: {P.DEVICE}")
    if P.DEVICE == "cpu":
        print("  (no GPU found — this will be very slow; a >=40GB GPU is recommended)")
    P.enable_determinism(args.seed)
    print("loading model + autoencoder (downloading weights on first run)...")
    model, diff, ae = P.load_models(age_conditioned=not args.base)

    for age in ages:
        tag = "atlas" if age is None else f"age{int(age):02d}"
        print(f"recovering {tag} ...", flush=True)
        vol = P.recover(model, diff, ae, age=age, cls=args.cls,
                        tstar=args.tstar, seed=args.seed, cfg_scale=args.cfg)
        path = P.save(vol, f"{args.out}/{tag}")
        print(f"  saved {path}  (+ preview {args.out}/{tag}.png)")
    print("done.")


if __name__ == "__main__":
    main()

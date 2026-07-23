#!/usr/bin/env python3
"""Recover a population atlas from a pretrained diffusion model.

    python recover.py                      # T1 brain atlas (base model)  -> outputs/atlas.nii.gz
    python recover.py --class 4            # a different anatomy (base model)
    python recover.py --age 70             # age-conditioned brain atlas
    python recover.py --ages 20,40,60,80   # a whole age family

By default this recovers the sharp T1 brain atlas from the original pretrained
generator (the paper's main result). `--age/--ages` switch to the age-conditioned
model, which yields one atlas per age (softer, fine-tuned population averages).
Each atlas is written as a NIfTI volume plus a PNG preview.
"""
import argparse
from atlas import pipeline as P


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--age", type=float, help="single age in years (age-conditioned model)")
    g.add_argument("--ages", help="comma-separated ages, e.g. 20,40,60,80 (age-conditioned model)")
    ap.add_argument("--class", dest="cls", type=int, default=P.CLASS_BRAIN_T1,
                    help=f"anatomy-class token for the base model (default {P.CLASS_BRAIN_T1}=brain-T1)")
    ap.add_argument("--out", default="outputs", help="output directory (default: outputs/)")
    ap.add_argument("--seed", type=int, default=11,
                    help="optional random seed; leave unset to draw a fresh x_T each run")
    ap.add_argument("--tstar", type=int, default=99, help="early-stopping time")
    ap.add_argument("--cfg", type=float, default=1.0, help="age guidance scale (>1)")
    args = ap.parse_args()

    age_mode = args.age is not None or args.ages is not None
    if args.ages:
        ages = [float(a) for a in args.ages.split(",")]
    elif args.age is not None:
        ages = [args.age]
    else:
        ages = [None]                              # base model, single anatomy atlas

    print(f"device: {P.DEVICE}")
    if P.DEVICE == "cpu":
        print("  (no GPU found — this will be very slow; a >=40GB GPU is recommended)")
    if args.seed is not None:
        P.enable_determinism(args.seed)
    which = "age-conditioned" if age_mode else "base"
    print(f"loading {which} model + autoencoder (downloading weights on first run)...")
    model, diff, ae = P.load_models(age_conditioned=age_mode)

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

#!/usr/bin/env python3
"""CLI: MS2 spectrum file(s) -> ranked candidate structures as JSON.

This is the container entry point. It loads MIST + MS-BART once, runs every
spectrum in the given file(s), and prints a JSON array to stdout (or --output).

Usage:
    python predict_cli.py INPUT.ms [INPUT2.mgf ...] [options]

Supported inputs:
    *.ms   SIRIUS single-spectrum file (carries >formula / >ionization headers)
    *.mgf  one or many spectra (needs FORMULA= and ADDUCT= per BEGIN IONS block)

Output (one object per spectrum, aligned to input order):
    [
      {"name": "...", "candidates": [
          {"rank": 1, "smiles": "...", "selfies": "...",
           "formula": "C15H10O4", "formula_diff": 0}, ...]},
      ...
    ]
"""
import argparse
import json
import sys
from pathlib import Path

from msbart_predict import MSBartPredictor, parse_mgf_file, parse_ms_file


def load_specs(paths):
    """Parse every input file into a flat list of Spectrum objects."""
    specs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"[warn] not found, skipping: {p}", file=sys.stderr)
            continue
        suffix = p.suffix.lower()
        if suffix == ".ms":
            s = parse_ms_file(p)
            if s is None:
                print(f"[warn] no usable peaks in {p}", file=sys.stderr)
                continue
            specs.append(s)
        elif suffix == ".mgf":
            file_specs = parse_mgf_file(p)
            if not file_specs:
                print(f"[warn] no spectra parsed from {p}", file=sys.stderr)
            specs.extend(file_specs)
        else:
            print(f"[warn] unsupported file type {suffix}, skipping: {p}", file=sys.stderr)
    return specs


def main():
    ap = argparse.ArgumentParser(
        description="MS-BART: MS2 spectrum -> ranked candidate molecular structures."
    )
    ap.add_argument("inputs", nargs="+", help="one or more .ms / .mgf spectrum files")
    ap.add_argument("-o", "--output", help="write JSON here instead of stdout")
    ap.add_argument("--device", default="auto",
                    help="auto|cuda|mps|cpu (default: auto -> CUDA, else MPS, else CPU)")
    ap.add_argument("--num-beams", type=int, default=10,
                    help="beam search width (default 10; raise to 50-100 on GPU)")
    ap.add_argument("--topk", type=int, default=10, help="candidates returned per spectrum")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--threshold", type=float, default=0.11,
                    help="fingerprint-probability cutoff for <fpXXXX> tokens")
    ap.add_argument("--no-formula-rerank", action="store_true",
                    help="skip molecular-formula re-ranking of beam candidates")
    args = ap.parse_args()

    specs = load_specs(args.inputs)
    if not specs:
        print("[error] no valid spectra to process", file=sys.stderr)
        sys.exit(2)

    predictor = MSBartPredictor(
        device=args.device,
        num_beams=args.num_beams,
        temperature=args.temperature,
        topk=args.topk,
        threshold=args.threshold,
    )
    print(f"[info] device={predictor.device}  spectra={len(specs)}  "
          f"beams={args.num_beams}", file=sys.stderr)

    results = predictor.predict(specs, use_formula_rerank=not args.no_formula_rerank)

    payload = [
        {"name": spec.name, "candidates": cands}
        for spec, cands in zip(specs, results)
    ]

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text)
        print(f"[info] wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()

"""Minimal usage example for msbart_predict.

Run from the `deploy/` directory (or add it to PYTHONPATH):
    python example.py path/to/spectrum.ms
    python example.py            # falls back to a built-in demo peak list
"""
import sys
from pprint import pprint

from msbart_predict import MSBartPredictor, parse_ms_file

# Load both models once. device="auto" -> CUDA, else Apple MPS, else CPU.
predictor = MSBartPredictor(device="auto", num_beams=10, topk=10)
print("device:", predictor.device)

if len(sys.argv) > 1:
    spec = parse_ms_file(sys.argv[1])
    candidates = predictor.predict([spec])[0]
else:
    # Demo: chrysin (C15H10O4), [M+H]+ — peaks abbreviated for illustration.
    candidates = predictor.predict_one(
        peaks=[[153.0188, 312.0], [163.0395, 420.0], [219.0653, 180.0],
               [255.0652, 999.0], [255.0653, 870.0]],
        formula="C15H10O4",
        adduct="[M+H]+",
        precursor_mz=255.065,
        name="demo",
    )

print(f"\nTop {len(candidates)} candidates:")
for c in candidates:
    print(f"  #{c['rank']:2d}  Δformula={c['formula_diff']}  {c['formula']:14s} {c['smiles']}")

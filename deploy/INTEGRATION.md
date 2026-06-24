# Embedding MS-BART into another Python program

This guide is for attaching MS-BART (MS2 spectrum → ranked candidate structures)
to an **existing Python** MS2-identification / report-generation program by
importing `msbart_predict` directly — no Docker, no HTTP, no subprocess.

```
your report program  ──import──▶  msbart_predict.MSBartPredictor
                                     ├─ MIST      (spectrum → fingerprint)
                                     └─ MS-BART   (fingerprint → SELFIES → SMILES)
```

The models load once into your process and stay resident, so per-spectrum calls
are cheap (no 457 MB reload).

---

## 1. One-time setup on the target machine

### 1a. Copy the bundle
Copy the whole `deploy/` folder to the target machine (anywhere, e.g.
`/opt/msbart/deploy`). It is self-contained except for the model weights
(downloaded in step 1c). `vendor/mist/` ships the MIST source, so no separate
MIST clone/install is needed.

### 1b. Environment + dependencies
```bash
cd /opt/msbart/deploy
python -m venv .venv && source .venv/bin/activate     # or conda; Python 3.9–3.11

# Install the torch build that matches THIS machine FIRST:
#   Linux + NVIDIA CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu124
#   Linux/macOS CPU (macOS gets Apple-MPS automatically):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install this package (editable keeps weights/ and vendor/ resolvable):
pip install -e .
```

> Why editable (`-e`)? The default weight/MIST paths are resolved relative to
> this folder. A non-editable install would move `msbart_predict` into
> site-packages and lose `weights/` and `vendor/`. If you must do a regular
> install, set `MSBART_WEIGHTS`, `MIST_CKPT`, `MIST_SRC` env vars to absolute
> paths instead.

### 1c. Download the model weights
The weights live on Google Drive (≈457 MB). `gdown` is pulled in by step 1b.
```bash
pip install gdown        # if not already present
bash scripts/fetch_weights.sh          # uses the bundled Google Drive link
# or override with a different archive:
# bash scripts/fetch_weights.sh "https://drive.google.com/uc?id=<file-id>"
```
This populates `weights/msbart/` and `weights/mist.ckpt`.

### 1d. Smoke test
```bash
python example.py                 # built-in demo spectrum
python predict_cli.py some.ms     # JSON candidates for a real file
```

---

## 2. Call it from your program

### Minimal
```python
from msbart_predict import MSBartPredictor

predictor = MSBartPredictor(device="auto")   # CUDA → MPS → CPU; load once at startup

candidates = predictor.predict_one(
    peaks=[[153.0188, 312.0], [255.0652, 999.0]],  # [m/z, intensity] pairs
    formula="C15H10O4",          # REQUIRED precursor neutral formula
    adduct="[M+H]+",
    precursor_mz=255.065,
    name="sample_001",
)
for c in candidates:             # ranked best-first
    print(c["rank"], c["smiles"], c["formula"], c["formula_diff"])
```

Each candidate is `{"rank", "smiles", "selfies", "formula", "formula_diff"}`.
`formula_diff` = atom-count difference vs the precursor formula
(`0` ⇒ candidate matches the known formula — usually your best hits).

### Recommended: a lazy singleton in your app
Model load (~2 s) should happen once, not per request.
```python
# msbart_singleton.py
from functools import lru_cache
from msbart_predict import MSBartPredictor

@lru_cache(maxsize=1)
def get_predictor():
    return MSBartPredictor(device="auto", num_beams=10, topk=10)
```
```python
# in your report code
from msbart_singleton import get_predictor

def annotate_spectrum(peaks, formula, adduct, precursor_mz, spec_id):
    cands = get_predictor().predict_one(
        peaks=peaks, formula=formula, adduct=adduct,
        precursor_mz=precursor_mz, name=spec_id,
    )
    return cands[: 5]   # surface several candidates in the report, not just #1
```

### Batch (faster than one-by-one)
```python
from msbart_predict import Spectrum
specs = [Spectrum(name=i, peaks=pk, formula=f, adduct=a) for ...]
results = predictor.predict(specs)     # list aligned to `specs`
```

### Thread-safety
A single predictor is **not** safe for concurrent `predict()` calls (shared
torch model state). For a multi-threaded/async server, either serialize calls
with a lock, or hold one predictor per worker process. Sequential use is fine.

---

## 3. Inputs you must provide

| field | required | notes |
|-------|----------|-------|
| `peaks` | ✅ | list/array of `[m/z, intensity]` |
| `formula` | ✅ | precursor **neutral** molecular formula (e.g. `C15H10O4`). MIST needs it to decompose fragments into sub-formulae; this matches how MS-BART was evaluated. Your pipeline (SIRIUS / formula annotation) already produces this. |
| `adduct` | ✅ | e.g. `[M+H]+`, `[M+Na]+`, `[M-H2O+H]+` |
| `precursor_mz` | optional | improves sub-formula assignment |
| `name` | optional | id echoed back in results |

SIRIUS `.ms` files already carry formula/adduct in their headers and parse
automatically via `predictor.predict_file("x.ms")`.

---

## 4. Knobs

| param | default | effect |
|-------|---------|--------|
| `device` | `"auto"` | `"auto"`/`"cuda"`/`"mps"`/`"cpu"` |
| `num_beams` | `10` | more beams = better recall, slower. GPU can do 50–100 |
| `temperature` | `0.4` | generation sampling temperature |
| `topk` | `10` | candidates returned (de-duplicated by SMILES) |
| `threshold` | `0.11` | fingerprint-probability cutoff for `<fpXXXX>` tokens |

Performance (measured): model load ~2 s; **CPU beam=10 ≈ 1.2 s/spectrum**;
GPU faster and comfortable at beam=100. Use beam=10 for interactive reporting.

Accuracy reference (cross-dataset SOP, beam=100): Top-1 Tanimoto 0.591,
Top-1 exact 22.97 %, Top-10 exact 44.59 % — so always show several candidates.

---

## 5. Troubleshooting

| symptom | fix |
|---------|-----|
| `Could not import 'mist'` | run from an editable install, or set `MIST_SRC` to the dir containing `vendor/mist` |
| weights not found | run `scripts/fetch_weights.sh`, or set `MSBART_WEIGHTS` / `MIST_CKPT` |
| empty candidate list for a spectrum | MIST dropped it (no usable sub-formulae) — check the formula/adduct are correct |
| slow on CPU with high beams | lower `num_beams` to 10 for interactive use |
| CUDA out of memory | lower `num_beams`, or `device="cpu"` |

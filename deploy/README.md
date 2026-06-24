# msbart_predict — portable MS-BART structure-elucidation module

Self-contained Python module that turns an **MS2 spectrum (+ precursor formula)**
into **ranked candidate molecular structures**, for embedding MS-BART into an
existing MS2 identification / report-generation program.

Pipeline:

```
MS2 spectrum + precursor formula + adduct
   → MIST              (spectrum → 4096-d fingerprint probabilities)
   → fingerprint tokens (prob > threshold → <fpXXXX> tokens)
   → MS-BART beam search (tokens → SELFIES candidates)
   → molecular-formula re-ranking
   → top-N candidate SMILES
```

## What's in this bundle

```
deploy/
├─ msbart_predict/        # the importable package
├─ vendor/mist/           # vendored MIST source (no separate install needed)
├─ weights/
│  ├─ msbart/             # MS-BART model + tokenizer (~399 MB)
│  └─ mist.ckpt           # MIST fingerprint model (~58 MB)
├─ requirements-infer.txt # slim inference deps (no vllm/deepspeed/ray/wandb)
├─ example.py
└─ README.md
```

To move it to another machine, **clone the repo** (the bundle is `deploy/`).
Put it at a **fixed** location and **next to** the program you're attaching it
to (not inside it), since the editable install below resolves `weights/` and
`vendor/` relative to this folder — don't move or delete it afterwards.

```bash
cd ~/work          # or /opt — somewhere stable, beside your program
git clone git@github.com:chalbori/MS-BART.git
# -> bundle at ~/work/MS-BART/deploy
```

## Install on the target machine (Ubuntu or macOS)

**Attaching to an existing Python program?** Activate *that program's* env
(don't make a new one) and install editable by path — then `import
msbart_predict` works from inside your program. Full guide: **INTEGRATION.md**.

```bash
# 0. (attach mode) activate your program's existing env:
#      source /path/to/your-program/.venv/bin/activate    # or: conda activate <env>
# (standalone testing only) make a throwaway env instead:
#      cd ~/work/MS-BART/deploy && python -m venv .venv && source .venv/bin/activate

# 1. Install PyTorch for the target platform FIRST
#    Linux + CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu124
#    Linux/macOS CPU (macOS automatically gets Apple-MPS support):
#    pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install this bundle editable (pulls the slim deps from pyproject.toml)
pip install -e ~/work/MS-BART/deploy

# 3. Download the model weights (~457 MB) from Google Drive
pip install gdown
bash ~/work/MS-BART/deploy/scripts/fetch_weights.sh   # -> weights/msbart/ + weights/mist.ckpt
```

> The weights are **not** in git. `fetch_weights.sh` pulls them from Google
> Drive (link baked into the script) and unpacks them into `weights/`.
>
> No-install alternative: skip step 2 and instead `pip install -r
> requirements-infer.txt`, then add the bundle to `sys.path`/`PYTHONPATH`
> (`msbart_predict` auto-adds `vendor/` for the MIST import).

## Use it (Python API)

```python
from msbart_predict import MSBartPredictor   # works directly after `pip install -e`
# (no install? -> import sys; sys.path.insert(0, "/path/to/deploy") first)

predictor = MSBartPredictor(device="auto")   # CUDA → MPS → CPU; loads models once

candidates = predictor.predict_one(
    peaks=[[153.0188, 312.0], [255.0652, 999.0], ...],  # [m/z, intensity]
    formula="C15H10O4",       # REQUIRED: precursor neutral formula
    adduct="[M+H]+",
    precursor_mz=255.065,
    name="sample_001",
)
for c in candidates:
    print(c["rank"], c["smiles"], c["formula"], c["formula_diff"])
```

`predict_one(...)` returns a ranked list of dicts:
`{"rank", "smiles", "selfies", "formula", "formula_diff"}`.
`formula_diff` is the atom-count difference vs the precursor formula
(0 = candidate matches the known formula).

### Other entry points

```python
# Batch of spectra (faster than one-by-one):
from msbart_predict import Spectrum
specs = [Spectrum(name="s1", peaks=[...], formula="C9H11NO2", adduct="[M+H]+"), ...]
results = predictor.predict(specs)          # list aligned to `specs`

# Directly from a file:
predictor.predict_file("sample.ms")          # SIRIUS .ms (single spectrum)
predictor.predict_file("batch.mgf")          # MGF (one or many spectra)
```

## Important: the precursor formula is required

MIST decomposes fragment peaks into **sub-formulae of the precursor**, so the
neutral molecular formula **must** be supplied (this matches how MS-BART was
evaluated). Your identification program almost certainly already produces a
formula (e.g. from SIRIUS / formula annotation) — pass it through. SIRIUS `.ms`
files carry it in the `>formula` header and are read automatically.

If you ever want to skip formula re-ranking (e.g. formula unknown), call
`predictor.predict(specs, use_formula_rerank=False)` — but the formula is still
needed for the MIST stage itself.

## Knobs

| param | default | effect |
|-------|---------|--------|
| `device` | `"auto"` | `"auto"`/`"cuda"`/`"mps"`/`"cpu"` |
| `num_beams` | `10` | more beams = better recall, slower. Paper uses up to 100 |
| `temperature` | `0.4` | sampling temperature for generation |
| `topk` | `10` | number of candidates returned (de-duplicated by SMILES) |
| `threshold` | `0.11` | fingerprint-probability cutoff for `<fpXXXX>` tokens |

## Performance (measured)

- Model load (both models): ~2 s
- **CPU, beam=10: ~1.2 s / spectrum**
- CUDA is faster and supports beam=100 comfortably.

beam=10 is recommended for an interactive reporting pipeline; raise to 50–100
for best-recall batch runs if you have a GPU.

## Accuracy reference (cross-dataset, beam=100)

On the SOP set (MS-BART trained on MassSpecGym): Top-1 Tanimoto 0.591,
Top-1 exact 22.97%, Top-10 exact 44.59%. So always surface several candidates
in the report, not just rank 1.

"""
SOP dataset preparation for MS-BART evaluation.

Pipeline:
  1. Read SOP .ms files (test split) → MGF + labels.tsv
  2. Run MIST fingerprint prediction
  3. Create MS-BART input TSV (fps, selfies, formula, ...)
"""

import sys
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import selfies as sf
from rdkit import Chem

import mist.subformulae.assign_subformulae as assign_subformulae
import mist.models.base as base
import mist.data.datasets as datasets
import mist.data.featurizers as featurizers

# ── Paths ────────────────────────────────────────────────────────────────────
SOP_SPEC_DIR   = Path("/home/swlee/Dev_others/DiffMS/data/sop/spec_files")
SOP_LABELS     = Path("/home/swlee/Dev_others/DiffMS/data/sop/labels.tsv")
SOP_SPLIT      = Path("/home/swlee/Dev_others/DiffMS/data/sop/split.tsv")

OUT_DIR        = Path("./data/SOP")
MGF_PATH       = OUT_DIR / "SOP.mgf"
LABELS_PATH    = OUT_DIR / "SOP_labels.tsv"
MIST_CKPT      = Path("./data/MassSpecGym/mist/mist.ckpt")
MIST_RES_DIR   = OUT_DIR / "mist"

THRESHOLD      = 0.11   # same as MassSpecGym eval

SUPPORTED_ADDUCTS = {
    "[M+H]+", "[M+Na]+", "[M+K]+",
    "[M-H2O+H]+", "[M+H3N+H]+", "[M]+", "[M-H4O2+H]+"
}


# ── Step 1: .ms → MGF + labels.tsv ──────────────────────────────────────────

def parse_ms_file(path: Path):
    """Parse a SIRIUS .ms file into a dict with meta and peaks."""
    meta = {}
    peaks = []
    in_peaks = False
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">ms2peaks") or line.startswith(">ms1peaks"):
                in_peaks = True
                continue
            if line.startswith(">") and in_peaks:
                in_peaks = False
            if in_peaks:
                parts = line.split()
                if len(parts) == 2:
                    peaks.append((float(parts[0]), float(parts[1])))
                continue
            if line.startswith(">"):
                kv = line[1:].split(None, 1)
                if len(kv) == 2:
                    meta[kv[0].lower()] = kv[1]
    if not peaks:
        return None
    return {"meta": meta, "peaks": np.array(peaks)}


def build_mgf_entry(spec_id: str, meta: dict, peaks: np.ndarray) -> str:
    lines = ["BEGIN IONS"]
    parentmass = meta.get("parentmass", "")
    if parentmass:
        lines.append(f"PEPMASS={parentmass}")
    lines.append(f"FEATURE_ID={spec_id}")
    adduct = meta.get("ionization", "[M+H]+")
    lines.append(f"ADDUCT={adduct}")
    lines.append(f"adduct={adduct}")
    formula = meta.get("formula", "")
    if formula:
        lines.append(f"FORMULA={formula}")
    for mz, intensity in sorted(peaks, key=lambda x: x[0]):
        lines.append(f"{mz} {intensity}")
    lines.append("END IONS")
    return "\n".join(lines)


def prepare_mgf_and_labels(split: str = "test"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_df  = pd.read_csv(SOP_SPLIT, sep="\t")
    labels_df = pd.read_csv(SOP_LABELS, sep="\t")

    test_ids = set(split_df[split_df["split"] == split]["name"].tolist())
    labels_df = labels_df[labels_df["spec"].isin(test_ids)].copy()
    labels_df = labels_df[labels_df["ionization"].isin(SUPPORTED_ADDUCTS)]

    mgf_entries = []
    label_rows = []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Parsing .ms files"):
        spec_id = row["spec"]
        ms_path = SOP_SPEC_DIR / f"{spec_id}.ms"
        if not ms_path.exists():
            print(f"  missing: {ms_path}")
            continue

        parsed = parse_ms_file(ms_path)
        if parsed is None:
            print(f"  no peaks: {spec_id}")
            continue

        meta = parsed["meta"]
        meta["ionization"] = row["ionization"]
        meta["formula"] = row["formula"]
        mgf_entries.append(build_mgf_entry(spec_id, meta, parsed["peaks"]))

        label_rows.append({
            "spec":        spec_id,
            "formula":     row["formula"],
            "ionization":  row["ionization"],
            "dataset":     "SOP",
            "compound":    spec_id,
            "parentmass":  float(meta.get("parentmass", 0)),
            "instrument":  row.get("instrument", "unknown"),
            "smiles":      row.get("smiles", ""),
            "inchikey":    row.get("inchikey", ""),
        })

    with open(MGF_PATH, "w") as f:
        f.write("\n\n".join(mgf_entries))
    print(f"MGF written: {MGF_PATH}  ({len(mgf_entries)} spectra)")

    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(LABELS_PATH, sep="\t", index=False)
    print(f"Labels written: {LABELS_PATH}")
    return label_df


# ── Step 2: MIST fingerprint prediction ─────────────────────────────────────

def run_mist():
    MIST_RES_DIR.mkdir(parents=True, exist_ok=True)
    subform_dir = MIST_RES_DIR / "subforms_fp"
    subform_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    fp_model = torch.load(MIST_CKPT, map_location=device)
    main_hparams = fp_model["hyper_parameters"]
    kwargs = copy.deepcopy(main_hparams)
    kwargs.update({
        "device": "cuda:0",
        "num_workers": 0,
        "subform_folder": str(subform_dir),
        "labels_file": str(LABELS_PATH),
    })

    model = base.build_model(**kwargs)
    model.load_state_dict(fp_model["state_dict"])
    model = model.to(device).eval()

    print("Assigning subformulae...")
    assign_subformulae.assign_subforms(
        spec_files=str(MGF_PATH),
        labels_file=str(LABELS_PATH),
        output_dir=str(subform_dir),
        mass_diff_thresh=20,
        max_formulae=50,
        num_workers=16,
        feature_id="FEATURE_ID",
        debug=False,
    )

    print("Predicting fingerprints...")
    kwargs["spec_features"] = model.spec_features(mode="test")
    kwargs["mol_features"] = "none"
    kwargs["allow_none_smiles"] = True
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)
    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))
    test_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=spectra_mol_pairs, featurizer=paired_featurizer, **kwargs
    )

    output_preds = model.encode_all_spectras(
        test_dataset, no_grad=True, **kwargs
    ).cpu().numpy()
    output_names = test_dataset.get_spectra_names()
    print(f"Fingerprint predictions shape: {output_preds.shape}")
    return output_preds, output_names


# ── Step 3: Build MS-BART input TSV ─────────────────────────────────────────

def build_msbart_tsv(output_preds, output_names, label_df):
    name_fps = {
        name: np.where(row > THRESHOLD)[0].tolist()
        for name, row in zip(output_names, output_preds)
    }

    label_df = label_df.set_index("spec")
    rows = []
    skipped = 0
    for spec_id, fps_indices in name_fps.items():
        if spec_id not in label_df.index:
            skipped += 1
            continue
        row = label_df.loc[spec_id]
        smiles = row.get("smiles", "")
        if not smiles:
            skipped += 1
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            selfies_str = sf.encoder(canonical_smiles)
        except Exception:
            skipped += 1
            continue

        fps_str = "".join([f"<fp{fp:04d}>" for fp in fps_indices])
        rows.append({
            "name":      spec_id,
            "fps":       fps_str,
            "selfies":   selfies_str,
            "smiles":    smiles,
            "inchikey":  row.get("inchikey", ""),
            "formula":   row.get("formula", ""),
            "adduct":    row.get("ionization", ""),
            "split":     "test",
        })

    df_out = pd.DataFrame(rows)
    tsv_path = OUT_DIR / f"test/SOP_fps_selfies_threshold_{THRESHOLD}.tsv"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(tsv_path, sep="\t", index=False)
    print(f"MS-BART input TSV: {tsv_path}  ({len(df_out)} entries, {skipped} skipped)")
    return tsv_path


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 1: .ms → MGF + labels.tsv ===")
    label_df = prepare_mgf_and_labels(split="test")

    print("\n=== Step 2: MIST fingerprint prediction ===")
    output_preds, output_names = run_mist()

    print("\n=== Step 3: Build MS-BART input TSV ===")
    tsv_path = build_msbart_tsv(output_preds, output_names, label_df)

    print(f"\nDone. Run eval with:\n"
          f"  conda activate ms-bart && "
          f"  accelerate launch src/eval_mp_post.py "
          f"--model_path data/MassSpecGym/model-wegihts "
          f"--test_path {tsv_path} "
          f"--num_beams 100 --temperature 0.4 --compute_mces")

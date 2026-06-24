"""Chain mist-cf formula/adduct inference into MS-BART structure prediction.

MS-BART *requires* a neutral precursor formula + adduct (MIST decomposes peaks
into sub-formulae of the precursor). When you don't have those, mist-cf
(github.com/samgoldman97/mist-cf, same author as MIST) infers a ranked list of
(neutral formula, adduct) hypotheses straight from the MS/MS spectrum.

This module bridges the two:

    mist-cf `formatted_output.tsv`  +  the MGF that was fed to mist-cf
        -> for each spectrum, take the top-N (formula, adduct) hypotheses
        -> run MS-BART once per hypothesis
        -> merge into one ranked candidate-structure list, each candidate
           tagged with the formula hypothesis it came from.

Typical use (after running `quickstart/run_model.sh` in the mist-cf repo):

    from msbart_predict import MSBartPredictor
    from msbart_predict.formula_infer import predict_from_mist_cf

    predictor = MSBartPredictor(device="auto")
    results = predict_from_mist_cf(
        predictor,
        mgf_path="mist-cf/data/demo_specs.mgf",
        mist_cf_tsv="mist-cf/quickstart/mist_cf_out/formatted_output.tsv",
        top_n=3,
    )
    for spec_id, cands in results.items():
        for c in cands:
            print(spec_id, c["rank"], c["smiles"], c["formula_hypothesis"],
                  c["adduct"], round(c["formula_score"], 2))
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .spectrum import SUPPORTED_ADDUCTS, Spectrum


@dataclass
class FormulaCandidate:
    """One (neutral formula, adduct) hypothesis from mist-cf for a spectrum."""
    spec_id: str
    formula: str
    adduct: str
    score: float
    rank: int


# ── mist-cf output parsing ────────────────────────────────────────────────────
def parse_mist_cf_output(
    tsv_path: Union[str, Path],
    top_n: int = 3,
    restrict_to_supported: bool = True,
    supported_adducts: Sequence[str] = tuple(SUPPORTED_ADDUCTS),
) -> Dict[str, List[FormulaCandidate]]:
    """Parse mist-cf `formatted_output.tsv` into per-spectrum formula hypotheses.

    Columns expected: spec | cand_form | scores | cand_ion | parentmasses | rank.

    Returns {spec_id: [FormulaCandidate, ...]} sorted by mist-cf rank, truncated
    to the first `top_n` *kept* hypotheses. When `restrict_to_supported` is True,
    hypotheses whose adduct is outside SUPPORTED_ADDUCTS are dropped before the
    top_n cut (so you still get up to top_n usable ones).
    """
    import pandas as pd

    supported = set(supported_adducts)
    df = pd.read_csv(tsv_path, sep="\t")
    # mist-cf already emits rows in ascending rank, but sort defensively.
    df = df.sort_values(["spec", "rank"], kind="stable")

    out: Dict[str, List[FormulaCandidate]] = {}
    for spec_id, grp in df.groupby("spec", sort=False):
        kept: List[FormulaCandidate] = []
        for _, row in grp.iterrows():
            adduct = str(row["cand_ion"]).strip()
            if restrict_to_supported and adduct not in supported:
                continue
            kept.append(FormulaCandidate(
                spec_id=str(spec_id),
                formula=str(row["cand_form"]).strip(),
                adduct=adduct,
                score=float(row["scores"]),
                rank=int(row["rank"]),
            ))
            if len(kept) >= top_n:
                break
        out[str(spec_id)] = kept
    return out


# ── lenient peak reader (no formula required, unlike spectrum.parse_mgf_file) ──
def read_mgf_peaks(
    mgf_path: Union[str, Path],
    id_key: str = "FEATURE_ID",
) -> Dict[str, dict]:
    """Read peaks/precursor m/z from an MGF, keyed by `id_key` (e.g. FEATURE_ID).

    Unlike `spectrum.parse_mgf_file`, this does NOT require a FORMULA field —
    the whole point is that mist-cf supplies the formula. Returns
    {spec_id: {"peaks": np.ndarray (N,2), "precursor_mz": float|None}}.
    """
    mgf_path = Path(mgf_path)
    id_key_l = id_key.lower()
    out: Dict[str, dict] = {}
    meta: dict = {}
    peaks: List[List[float]] = []
    in_ions = False
    with open(mgf_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                in_ions, meta, peaks = True, {}, []
                continue
            if line == "END IONS":
                if peaks:
                    sid = (meta.get(id_key_l) or meta.get("title")
                           or f"spec_{len(out)}")
                    pm = meta.get("pepmass") or meta.get("parentmass")
                    pm_val = float(pm.split()[0]) if pm else None
                    out[str(sid)] = {
                        "peaks": np.array(peaks, dtype=float),
                        "precursor_mz": pm_val,
                    }
                in_ions = False
                continue
            if not in_ions:
                continue
            if "=" in line and not line[0].isdigit():
                k, v = line.split("=", 1)
                meta[k.strip().lower()] = v.strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    peaks.append([float(parts[0]), float(parts[1])])
    return out


# ── the chain ─────────────────────────────────────────────────────────────────
def _merge_candidates(per_hypothesis: List[List[dict]]) -> List[dict]:
    """Flatten per-hypothesis candidate lists into one ranked, de-duplicated list.

    Trusts mist-cf's formula ranking first (lower formula_rank = more confident
    formula), then MS-BART's own rank within a hypothesis. De-duplicates by
    SMILES, keeping the first (best) occurrence, then re-numbers `rank` globally.
    """
    flat: List[dict] = []
    for cands in per_hypothesis:
        flat.extend(cands)
    # stable sort: formula_rank asc, then the MS-BART rank within the hypothesis.
    flat.sort(key=lambda c: (c.get("formula_rank", 1_000), c.get("msbart_rank", 1_000)))

    seen = set()
    merged: List[dict] = []
    for c in flat:
        key = c.get("smiles")
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)
    for i, c in enumerate(merged, start=1):
        c["rank"] = i
    return merged


def predict_from_mist_cf(
    predictor,
    mgf_path: Union[str, Path],
    mist_cf_tsv: Union[str, Path],
    top_n: int = 3,
    id_key: str = "FEATURE_ID",
    restrict_to_supported: bool = True,
    merge: bool = True,
    **predict_kwargs,
) -> Dict[str, List[dict]]:
    """Run the full mist-cf -> MS-BART chain over a batch of spectra.

    Args:
        predictor: a constructed `MSBartPredictor`.
        mgf_path: the SAME MGF that was fed to mist-cf (provides the peaks).
        mist_cf_tsv: mist-cf's `formatted_output.tsv` (provides formula/adduct).
        top_n: number of formula hypotheses to try per spectrum (default 3).
        restrict_to_supported: drop hypotheses whose adduct MS-BART wasn't
            evaluated on (keeps up to top_n of the remaining).
        merge: if True, return one merged ranked list per spectrum; if False,
            return the raw per-hypothesis lists (each tagged with provenance).
        **predict_kwargs: forwarded to `predictor.predict` (num_beams, topk, ...).

    Returns {spec_id: [candidate dict, ...]}. Each candidate carries the usual
    MS-BART keys plus provenance: `formula_hypothesis`, `adduct`,
    `formula_score`, `formula_rank`, `msbart_rank`.
    """
    peaks_by_id = read_mgf_peaks(mgf_path, id_key=id_key)
    hyps_by_id = parse_mist_cf_output(
        mist_cf_tsv, top_n=top_n, restrict_to_supported=restrict_to_supported,
    )

    results: Dict[str, List[dict]] = {}
    for spec_id, hyps in hyps_by_id.items():
        if spec_id not in peaks_by_id:
            # mist-cf scored a spectrum we have no peaks for (id mismatch).
            results[spec_id] = []
            continue
        peaks = peaks_by_id[spec_id]["peaks"]
        precursor_mz = peaks_by_id[spec_id]["precursor_mz"]

        per_hypothesis: List[List[dict]] = []
        for h in hyps:
            try:
                spec = Spectrum(
                    name=f"{spec_id}__{h.formula}_{h.adduct}",
                    peaks=peaks, formula=h.formula, adduct=h.adduct,
                    precursor_mz=precursor_mz,
                )
                cands = predictor.predict([spec], **predict_kwargs)[0]
            except Exception:
                # A single bad hypothesis (e.g. MIST drops it) shouldn't sink
                # the others for this spectrum.
                cands = []
            for c in cands:
                c["msbart_rank"] = c.get("rank")
                c["formula_hypothesis"] = h.formula
                c["adduct"] = h.adduct
                c["formula_score"] = h.score
                c["formula_rank"] = h.rank
            per_hypothesis.append(cands)

        results[spec_id] = (
            _merge_candidates(per_hypothesis) if merge
            else [c for lst in per_hypothesis for c in lst]
        )
    return results

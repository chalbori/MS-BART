"""Spectrum representation, parsers (.ms / MGF / peak list), and MIST input writers."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PeakList = Sequence[Tuple[float, float]]

# Adducts MIST/MS-BART was evaluated with. Others still run but may be off-distribution.
SUPPORTED_ADDUCTS = {
    "[M+H]+", "[M+Na]+", "[M+K]+",
    "[M-H2O+H]+", "[M+H3N+H]+", "[M]+", "[M-H4O2+H]+",
}


@dataclass
class Spectrum:
    """A single MS2 spectrum ready for the MS-BART pipeline.

    formula (neutral precursor molecular formula) is REQUIRED: MIST decomposes
    peaks into sub-formulae of the precursor, so it cannot run without it.
    """
    name: str
    peaks: np.ndarray            # shape (N, 2): [m/z, intensity]
    formula: str
    adduct: str = "[M+H]+"
    precursor_mz: Optional[float] = None
    smiles: str = ""             # optional ground-truth, only used for eval
    inchikey: str = ""
    instrument: str = "unknown"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.peaks = np.asarray(self.peaks, dtype=float).reshape(-1, 2)
        if not self.formula:
            raise ValueError(
                f"Spectrum '{self.name}' has no precursor formula; "
                "MIST requires it for sub-formula assignment."
            )
        if self.peaks.shape[0] == 0:
            raise ValueError(f"Spectrum '{self.name}' has no peaks.")


def parse_ms_file(path: Union[str, Path]) -> Optional[Spectrum]:
    """Parse a SIRIUS .ms file into a Spectrum (formula/adduct read from header)."""
    path = Path(path)
    meta = {}
    peaks: List[Tuple[float, float]] = []
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
    parentmass = meta.get("parentmass") or meta.get("precursormz")
    return Spectrum(
        name=path.stem,
        peaks=np.array(peaks),
        formula=meta.get("formula", ""),
        adduct=meta.get("ionization", "[M+H]+"),
        precursor_mz=float(parentmass) if parentmass else None,
    )


def parse_mgf_file(path: Union[str, Path]) -> List[Spectrum]:
    """Parse an MGF file into Spectra. Reads FORMULA, ADDUCT, PEPMASS, FEATURE_ID."""
    path = Path(path)
    specs: List[Spectrum] = []
    meta = {}
    peaks: List[Tuple[float, float]] = []
    in_ions = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                in_ions, meta, peaks = True, {}, []
                continue
            if line == "END IONS":
                if peaks:
                    name = meta.get("feature_id") or meta.get("title") or f"spec_{len(specs)}"
                    pm = meta.get("pepmass", "").split()[0] if meta.get("pepmass") else None
                    specs.append(Spectrum(
                        name=str(name),
                        peaks=np.array(peaks),
                        formula=meta.get("formula", ""),
                        adduct=meta.get("adduct", "[M+H]+"),
                        precursor_mz=float(pm) if pm else None,
                    ))
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
                    peaks.append((float(parts[0]), float(parts[1])))
    return specs


def _build_mgf_entry(spec: Spectrum) -> str:
    lines = ["BEGIN IONS"]
    if spec.precursor_mz is not None:
        lines.append(f"PEPMASS={spec.precursor_mz}")
    lines.append(f"FEATURE_ID={spec.name}")
    lines.append(f"ADDUCT={spec.adduct}")
    lines.append(f"adduct={spec.adduct}")
    if spec.formula:
        lines.append(f"FORMULA={spec.formula}")
    for mz, intensity in sorted(spec.peaks.tolist(), key=lambda x: x[0]):
        lines.append(f"{mz} {intensity}")
    lines.append("END IONS")
    return "\n".join(lines)


def write_mist_inputs(specs: Sequence[Spectrum], out_dir: Union[str, Path]):
    """Write MGF + labels.tsv for MIST. Returns (mgf_path, labels_path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mgf_path = out_dir / "input.mgf"
    labels_path = out_dir / "labels.tsv"

    with open(mgf_path, "w") as f:
        f.write("\n\n".join(_build_mgf_entry(s) for s in specs))

    # NOTE: deliberately omit the `smiles`/`inchikey` columns. MIST only needs
    # them to attach ground-truth molecules (unused for fingerprint prediction),
    # and an absent column makes MIST build empty placeholder Mols. If we wrote
    # an empty/NaN smiles cell instead, pandas.astype(str) turns it into the
    # string "nan", which RDKit fails to parse -> crash.
    rows = []
    for s in specs:
        rows.append({
            "spec": s.name,
            "formula": s.formula,
            "ionization": s.adduct,
            "dataset": "DEPLOY",
            "compound": s.name,
            "parentmass": float(s.precursor_mz) if s.precursor_mz is not None else 0.0,
            "instrument": s.instrument,
        })
    pd.DataFrame(rows).to_csv(labels_path, sep="\t", index=False)
    return mgf_path, labels_path

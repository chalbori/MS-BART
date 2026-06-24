"""Top-level MS-BART predictor: raw spectrum -> ranked candidate structures.

Usage:
    from msbart_predict import MSBartPredictor
    predictor = MSBartPredictor()                 # loads both models once
    cands = predictor.predict_one(
        peaks=[[101.07, 1234.0], [130.06, 980.0], ...],
        formula="C9H11NO2",
        adduct="[M+H]+",
        precursor_mz=166.08,
        name="sample1",
    )
    # cands -> [{rank, smiles, selfies, formula, formula_diff}, ...]
"""
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from . import config
from .device import pick_device
from .mist_fp import MISTFingerprinter
from .msbart_gen import MSBartGenerator
from .spectrum import Spectrum, parse_mgf_file, parse_ms_file


class MSBartPredictor:
    def __init__(
        self,
        msbart_weights=None,
        mist_ckpt=None,
        mist_src=None,
        device: str = "auto",
        threshold: float = 0.11,
        num_beams: int = 10,
        temperature: float = 0.4,
        topk: int = 10,
        mist_num_workers: Optional[int] = None,
    ):
        config.ensure_mist_importable(mist_src)
        self.device = pick_device(device)
        self.threshold = threshold
        self.num_beams = num_beams
        self.temperature = temperature
        self.topk = topk

        # On CPU/MPS the subformula multiprocessing is fine but keep it modest by default.
        if mist_num_workers is None:
            mist_num_workers = 0 if self.device.type in ("cpu", "mps") else 8

        self.mist = MISTFingerprinter(
            config.resolve_mist_ckpt(mist_ckpt),
            device=self.device,
            num_workers=mist_num_workers,
        )
        self.msbart = MSBartGenerator(
            config.resolve_msbart_weights(msbart_weights),
            device=self.device,
        )

    # ── batch API ────────────────────────────────────────────────────────────
    def predict(
        self,
        specs: Sequence[Spectrum],
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        use_formula_rerank: bool = True,
    ) -> List[List[dict]]:
        """Predict candidates for a batch of Spectrum objects.

        Returns a list (aligned to `specs`) of ranked candidate lists.
        """
        num_beams = num_beams or self.num_beams
        temperature = temperature if temperature is not None else self.temperature
        topk = topk or self.topk

        fp_preds, names = self.mist.predict(specs)
        # MIST may reorder; map back to the input order by name.
        name_to_row = {n: fp_preds[i] for i, n in enumerate(names)}

        ordered_specs, fps_tokens, formulas = [], [], []
        for s in specs:
            if s.name not in name_to_row:
                # MIST dropped this spectrum (e.g. no usable sub-formulae)
                ordered_specs.append(s)
                fps_tokens.append("")
                formulas.append(s.formula)
                continue
            ordered_specs.append(s)
            fps_tokens.append(self.mist.fps_to_tokens(name_to_row[s.name], self.threshold))
            formulas.append(s.formula)

        gen = self.msbart.generate(
            fps_tokens,
            formulas=formulas if use_formula_rerank else None,
            num_beams=num_beams,
            temperature=temperature,
            topk=topk,
        )
        return gen

    # ── convenience single-spectrum API ──────────────────────────────────────
    def predict_one(
        self,
        peaks,
        formula: str,
        adduct: str = "[M+H]+",
        precursor_mz: Optional[float] = None,
        name: str = "spectrum",
        **kwargs,
    ) -> List[dict]:
        spec = Spectrum(
            name=name, peaks=np.asarray(peaks, dtype=float),
            formula=formula, adduct=adduct, precursor_mz=precursor_mz,
        )
        return self.predict([spec], **kwargs)[0]

    # ── file helpers ──────────────────────────────────────────────────────────
    def predict_file(self, path: Union[str, Path], **kwargs) -> List[List[dict]]:
        """Predict from a .ms (single) or .mgf (possibly many) file."""
        path = Path(path)
        if path.suffix.lower() == ".ms":
            spec = parse_ms_file(path)
            if spec is None:
                return []
            specs = [spec]
        elif path.suffix.lower() in (".mgf",):
            specs = parse_mgf_file(path)
        else:
            raise ValueError(f"Unsupported spectrum file type: {path.suffix}")
        return self.predict(specs, **kwargs)

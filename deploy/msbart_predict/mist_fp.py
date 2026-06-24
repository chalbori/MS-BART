"""MIST fingerprint stage: spectra -> predicted Morgan-fingerprint probabilities.

Adapted from preprocess/prepare_sop_for_msbart.run_mist, but:
  * loads the checkpoint once (reusable across calls),
  * auto-detects the device (CUDA / MPS / CPU),
  * works on a batch of in-memory Spectrum objects via a temp dir.
"""
import copy
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .spectrum import Spectrum, write_mist_inputs


class MISTFingerprinter:
    def __init__(self, mist_ckpt, device: torch.device, num_workers: int = 0):
        # Imported here so a missing MIST install fails loudly only when used.
        import mist.models.base as base  # noqa: F401

        self.device = device
        self.num_workers = num_workers
        self.mist_ckpt = str(mist_ckpt)

        ckpt = torch.load(self.mist_ckpt, map_location="cpu", weights_only=False)
        self.base_hparams = copy.deepcopy(ckpt["hyper_parameters"])
        kwargs = copy.deepcopy(self.base_hparams)
        kwargs.update({"device": str(device), "num_workers": num_workers})
        self.model = base.build_model(**kwargs)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(device).eval()

    def predict(self, specs: Sequence[Spectrum], work_dir: Optional[Path] = None):
        """Return (fingerprint_array [N, D], spectra_names) aligned to MIST output order."""
        import mist.subformulae.assign_subformulae as assign_subformulae
        import mist.data.datasets as datasets
        import mist.data.featurizers as featurizers

        tmp_ctx = None
        if work_dir is None:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="msbart_mist_")
            work_dir = Path(tmp_ctx.name)
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        try:
            mgf_path, labels_path = write_mist_inputs(specs, work_dir)
            subform_dir = work_dir / "subforms_fp"
            subform_dir.mkdir(parents=True, exist_ok=True)

            kwargs = copy.deepcopy(self.base_hparams)
            kwargs.update({
                "device": str(self.device),
                "num_workers": self.num_workers,
                "subform_folder": str(subform_dir),
                "labels_file": str(labels_path),
            })

            assign_subformulae.assign_subforms(
                spec_files=str(mgf_path),
                labels_file=str(labels_path),
                output_dir=str(subform_dir),
                mass_diff_thresh=20,
                max_formulae=50,
                num_workers=self.num_workers,
                feature_id="FEATURE_ID",
                debug=False,
            )

            kwargs["spec_features"] = self.model.spec_features(mode="test")
            kwargs["mol_features"] = "none"
            kwargs["allow_none_smiles"] = True
            paired_featurizer = featurizers.get_paired_featurizer(**kwargs)
            spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
            spectra_mol_pairs = list(zip(*spectra_mol_pairs))
            test_dataset = datasets.SpectraMolDataset(
                spectra_mol_list=spectra_mol_pairs, featurizer=paired_featurizer, **kwargs
            )

            preds = self.model.encode_all_spectras(
                test_dataset, no_grad=True, **kwargs
            ).cpu().numpy()
            names = test_dataset.get_spectra_names()
            return preds, names
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    @staticmethod
    def fps_to_tokens(fp_row: np.ndarray, threshold: float) -> str:
        """Convert a fingerprint probability row to MS-BART '<fpXXXX>' token string."""
        idxs = np.where(fp_row > threshold)[0].tolist()
        return "".join(f"<fp{i:04d}>" for i in idxs)

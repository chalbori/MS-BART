"""msbart_predict: portable raw-spectrum -> molecular-structure inference for MS-BART.

Pipeline:  MS2 spectrum (+ precursor formula/adduct)
             -> MIST fingerprint prediction
             -> MS-BART beam generation (SELFIES)
             -> molecular-formula re-ranking
             -> top-N candidate SMILES.
"""
from .predictor import MSBartPredictor
from .spectrum import Spectrum, parse_ms_file, parse_mgf_file
from .device import pick_device
from .formula_infer import predict_from_mist_cf, parse_mist_cf_output

__all__ = [
    "MSBartPredictor",
    "Spectrum",
    "parse_ms_file",
    "parse_mgf_file",
    "pick_device",
    "predict_from_mist_cf",
    "parse_mist_cf_output",
]

__version__ = "0.1.0"

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

__all__ = [
    "MSBartPredictor",
    "Spectrum",
    "parse_ms_file",
    "parse_mgf_file",
    "pick_device",
]

__version__ = "0.1.0"

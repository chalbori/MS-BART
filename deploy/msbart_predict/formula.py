"""Formula helpers used for SELFIES->formula conversion and beam re-ranking.

Lifted from src/eval_mp_post.py so the deploy bundle is self-contained.
"""
import re
from collections import defaultdict

from selfies import decoder
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def selfies_to_smiles(selfies: str):
    try:
        smiles = decoder(selfies)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def selfies_to_formula(selfies: str):
    try:
        smiles = decoder(selfies)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def _parse_formula(formula: str, ignore_h: bool = False):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
    counts = defaultdict(int)
    for elem, cnt in pattern.findall(formula):
        if not elem:
            continue
        if ignore_h and elem == "H":
            continue
        counts[elem] += int(cnt) if cnt else 1
    return counts


def compare_formulas(formula1: str, formula2: str, ignore_h: bool = False):
    """Return (per-element diff dict, total atom-count difference)."""
    d1 = _parse_formula(formula1, ignore_h)
    d2 = _parse_formula(formula2, ignore_h)
    diffs = {}
    diff_cnt = 0
    for elem in set(d1) | set(d2):
        diff = abs(d1.get(elem, 0) - d2.get(elem, 0))
        if diff:
            diffs[elem] = diff
            diff_cnt += diff
    return diffs, diff_cnt

"""MS-BART generation stage: fingerprint tokens -> ranked candidate structures.

Adapted from src/eval_mp_post.py, minus accelerate/MCES/label-dependence.
Generates N beam candidates, then re-ranks by molecular-formula agreement
with the (known) precursor formula -- the paper's strong inference setting.
"""
from typing import List, Optional, Sequence

import torch
from transformers import BartForConditionalGeneration, BartTokenizer, GenerationConfig

from .formula import compare_formulas, selfies_to_formula, selfies_to_smiles


class MSBartGenerator:
    def __init__(self, model_path, device: torch.device):
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
        self.model.eval()

    def generate(
        self,
        fps_tokens: Sequence[str],
        formulas: Optional[Sequence[str]] = None,
        num_beams: int = 10,
        temperature: float = 0.4,
        topk: int = 10,
        max_new_tokens: int = 256,
        batch_size: Optional[int] = None,
    ) -> List[List[dict]]:
        """For each input, return a ranked list (<=topk) of candidate dicts:
        {rank, selfies, smiles, formula, formula_diff}.

        If `formulas` is given, candidates are re-ranked by formula agreement;
        otherwise the raw beam order is kept.
        """
        if batch_size is None:
            batch_size = 4 if num_beams > 20 else 16

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_beams,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        results: List[List[dict]] = []
        for start in range(0, len(fps_tokens), batch_size):
            batch_fps = list(fps_tokens[start:start + batch_size])
            n = len(batch_fps)
            inputs = self.tokenizer(
                batch_fps, return_tensors="pt", padding=True,
                padding_side="right", add_special_tokens=False,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out_ids = self.model.generate(**inputs, generation_config=gen_cfg)
            out_ids = out_ids.cpu()
            decoded = [
                self.tokenizer.decode(x, skip_special_tokens=True).replace(" ", "")
                for x in out_ids
            ]
            # decoded is flat: n * num_beams; regroup per input
            grouped = [decoded[i * num_beams:(i + 1) * num_beams] for i in range(n)]

            for j, cands in enumerate(grouped):
                target_formula = formulas[start + j] if formulas is not None else None
                results.append(self._rank(cands, target_formula, topk))
        return results

    @staticmethod
    def _rank(candidate_selfies: Sequence[str], target_formula: Optional[str], topk: int):
        scored = []
        for idx, s in enumerate(candidate_selfies):
            f = selfies_to_formula(s)
            if f is None:
                continue
            diff = 0
            if target_formula:
                _, diff = compare_formulas(target_formula, f, ignore_h=False)
            scored.append({"selfies": s, "formula": f, "formula_diff": diff, "index": idx})

        scored.sort(key=lambda x: (x["formula_diff"], x["index"]))

        out = []
        seen = set()
        for item in scored:
            smi = selfies_to_smiles(item["selfies"])
            if smi is None or smi in seen:
                continue
            seen.add(smi)
            out.append({
                "rank": len(out) + 1,
                "selfies": item["selfies"],
                "smiles": smi,
                "formula": item["formula"],
                "formula_diff": item["formula_diff"],
            })
            if len(out) >= topk:
                break
        return out

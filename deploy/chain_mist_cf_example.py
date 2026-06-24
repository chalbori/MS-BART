"""End-to-end example: mist-cf (formula/adduct) -> MS-BART (structures).

Run mist-cf's quickstart first (produces formatted_output.tsv), then:

    python chain_mist_cf_example.py \
        --mgf ~/Dev_others/mist-cf/data/demo_specs.mgf \
        --tsv ~/Dev_others/mist-cf/quickstart/mist_cf_out/formatted_output.tsv \
        --top-n 3 --only CCMSLIB00000001590

Prints, per spectrum, the merged ranked candidate structures with the formula
hypothesis each came from.
"""
import argparse
import json

from msbart_predict import MSBartPredictor
from msbart_predict.formula_infer import predict_from_mist_cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgf", required=True, help="MGF that was fed to mist-cf")
    ap.add_argument("--tsv", required=True, help="mist-cf formatted_output.tsv")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--num-beams", type=int, default=10)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default=None, help="restrict to one spec id (faster)")
    ap.add_argument("--json", default=None, help="optional path to dump JSON")
    args = ap.parse_args()

    predictor = MSBartPredictor(device=args.device, num_beams=args.num_beams,
                                topk=args.topk)
    results = predict_from_mist_cf(
        predictor, mgf_path=args.mgf, mist_cf_tsv=args.tsv, top_n=args.top_n,
    )
    if args.only:
        results = {k: v for k, v in results.items() if k == args.only}

    for spec_id, cands in results.items():
        print(f"\n=== {spec_id} ({len(cands)} candidates) ===")
        for c in cands[:args.topk]:
            print(f"  #{c['rank']:<2} {c['smiles']:<45} "
                  f"hyp={c['formula_hypothesis']:<14} {c['adduct']:<11} "
                  f"fscore={c['formula_score']:.2f} frank={c['formula_rank']} "
                  f"Δform={c['formula_diff']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

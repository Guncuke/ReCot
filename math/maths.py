import argparse
import os

from datasets import load_dataset

from open_thoughts import decontaminate, deduplicate
from open_thoughts.math.filter import filter_problems
from open_thoughts.math.reason import reason

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    ds = load_dataset("/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/math_data/datasets--AI-MO--NuminaMath-CoT", split="train")
    ds = ds.filter(lambda x: x["source"] in ["amc_aime", "olympiads", "aops_forum", "math"])
    ds = ds.filter(filter_problems)
    ds = ds.rename_column("source", "source_subset")
    ds = ds.rename_column("problem", "question")
    ds = ds.add_column("domain", ["math"] * len(ds))
    ds = ds.add_column("source", ["numina_math"] * len(ds))

    if args.dry_run:
        ds = ds.take(12000)

    ds = deduplicate(ds)
    ds = decontaminate(ds)
    ds = reason(ds)

    if args.dry_run:
        print("======== MATH DATASET ========")
        print(ds)
        print(ds[0])
        print("================")

    ds.to_json("/public/data0/NLP/users/wucanhui.volcano/output/math_data.jsonl", orient="records", lines=True)

import argparse
import os

from open_thoughts.code.combine import combine
from open_thoughts.code.reason import reason
from open_thoughts.code.standardize import standardize
from open_thoughts.decontaminate import decontaminate
from open_thoughts.deduplicate import deduplicate
from open_thoughts.prompt import SKY_T1_SYSTEM_PROMPT, format_code_prompt


def map_code_to_share_gpt(row):
    user_message = format_code_prompt(row)
    assistant_message = (
        f"<|begin_of_thought|>\n\n{row['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{row['deepseek_solution']}\n\n<|end_of_solution|>"
    )

    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    # Each of the subsets below is formatted in a different way, we first standardize them
    # into a common format and combine them.
    # dry run only on apps

    if args.dry_run:
        subsets = {
            "/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/code_data/datasets--MatrixStudio--Codeforces-Python-Submissions": None,
            "/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/code_data/datasets--BAAI--TACO": None,
            "/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/code_data/datasets--codeparrot--apps": None,
        }
    else:
        subsets = {
            "/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/code_data/datasets--MatrixStudio--Codeforces-Python-Submissions": None,
            "/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/code_data/datasets--BAAI--TACO": None,
            "/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/code_data/datasets--codeparrot--apps": None,
            "/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/code_data/datasets--deepmind--code_contests": None,
        }

    for subset in subsets:
        print(f"Standardizing {subset}...")
        print(f"subset: {subset}")
        ds = standardize(subset, num_hf_proc_workers=os.cpu_count(), dry_run=args.dry_run)
        ds = ds.add_column("subset", [subset] * len(ds))

        if args.dry_run:
            if subset == "/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/code_data/datasets--MatrixStudio--Codeforces-Python-Submissions":
                subsets[subset] = ds.take(900)
            else:
                subsets[subset] = ds.take(3000)
        else:
            subsets[subset] = ds

    ds = combine(subsets, dry_run=args.dry_run)
    # Deduplicate and decontaminate the dataset against benchmarks.
    ds = deduplicate(ds, column="problem")
    ds = decontaminate(ds, column="problem")

    # Annotate the dataset with reasoning.
    ds = reason(ds)

    if args.dry_run:
        print("======== CODE DATASET ========")
        print(ds)
        print(ds[0])
        print("================")

    ds = ds.add_column("domain", ["code"] * len(ds))
    ds.to_json("/public/data0/NLP/users/wucanhui.volcano/output/code_data.jsonl", orient="records", lines=True)

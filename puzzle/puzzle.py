import argparse
import os
from datasets import load_dataset
from open_thoughts import decontaminate, deduplicate
from open_thoughts.puzzle.reason import reason


def riddle_sense_map(x):
    question = x["question"]
    choices = x["choices"]
    full_question = question
    labels = choices["label"]
    texts = choices["text"]
    for label, text in zip(labels, texts):
        full_question += f"\n{label}: {text}"
    return {"question": full_question, "answer": x["answerKey"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    ds = load_dataset("/public/data0/NLP/users/wucanhui.volcano/datasets/all_self_gen_data/INK-USC/riddle_sense", split="train", trust_remote_code=True)
    ds = ds.map(riddle_sense_map)

    if args.dry_run:
        ds = ds.take(3500)
    else:
        ds = ds.shuffle(seed=42).take(1_250)

    ds = ds.remove_columns(["answerKey", "choices"])
    ds = ds.add_column("domain", ["puzzle"] * len(ds))
    ds = ds.add_column("source", ["riddle_sense"] * len(ds))

    ds = deduplicate(ds)
    ds = decontaminate(ds)
    ds = reason(ds)

    if args.dry_run:
        print("======== PUZZLE DATASET ========")
        print(ds)
        print(ds[0])
        print("================")

    ds.to_json("/public/data0/NLP/users/wucanhui.volcano/output_yx/puzzle_data.jsonl", orient="records", lines=True)

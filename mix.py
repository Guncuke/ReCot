import argparse
import os
import platform
from multiprocessing import freeze_support
from datasets import concatenate_datasets, load_dataset
from open_thoughts import prompt, verify

parser = argparse.ArgumentParser()
args = parser.parse_args()

if __name__ == "__main__":
    # on a mac, freeze support
    if platform.system() == "Darwin":
        freeze_support()

    verified_mix_ds = []
 
    for subset in ["puzzle", "math", "code"]:
        ds = load_dataset('json', data_files=f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_data.jsonl", split='train')
        verified_ds, falsed_ds = verify(ds)

        if len(verified_ds) > 0:
            verified_ds.to_json(f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_verified_data.jsonl", orient="records", lines=True)
            verified_mix_ds.append(verified_ds)

    verified_mix = concatenate_datasets(verified_mix_ds)
    verified_mix.to_json("/public/data0/NLP/users/wucanhui.volcano/output/verified_mix.jsonl", orient="records", lines=True)

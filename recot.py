from datasets import load_dataset
from open_thoughts import verify
from open_thoughts import shorten

if __name__ == "__main__":

    iteration = 2
    for iter in range(iteration):
        for subset in ["math", "puzzle", "code"]:
            if iter == 0:
                ds = load_dataset('json', data_files=f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_verified_data.jsonl", split='train')
            else:
                ds = load_dataset('json', data_files=f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_shorten_data.jsonl", split='train')
            # 使用大模型，缩短reasoning长度
            ds = ds.take(1)
            ds = shorten(ds, 10)
            # ds = boost(ds)
            ds.to_json(f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_shorten_data.jsonl", orient="records", lines=True)

            # if len(verified_ds) > 0:
            #     verified_ds.to_json(f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_verified_data.jsonl", orient="records", lines=True)
    # verified_ds, falsed_ds = verify(ds)

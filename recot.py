from datasets import load_dataset, concatenate_datasets
from open_thoughts import verify, shorten, boost
from vllm import LLM
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



if __name__ == "__main__":

    iteration = 9
    llm = LLM(model="/public/data0/NLP/users/wucanhui.volcano/models/DeepSeek-R1-Distill-Qwen-7B", tensor_parallel_size=4)
    all_wrong_ds = []
    for iter in range(8, iteration):
        print(f"start iteration {iter}")
        for subset in ["math", "puzzle", "code"]:
            # 第一个iter，读取原始数据
            if iter == 0:
                ds = load_dataset('json', data_files=f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_verified_data.jsonl", split='train')
            # 最后一个iter，读取上一个iter的数据，新的reasnoning内直接删除思考过程
            elif iter == iteration - 1:
                ds = load_dataset('json', data_files=f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_shorten_iter{iter-1}_data.jsonl", split='train')
                ds = ds.map(lambda example: {'reasoning': example['shorten_reasoning']})
                ds = ds.map(lambda example: {'shorten_reasoning': ""})
            # 其他iter，读取上一个iter的数据，读取缩短的reasoning作为新的reasoning
            else:
                ds = load_dataset('json', data_files=f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_shorten_iter{iter-1}_data.jsonl", split='train')
                ds = ds.map(lambda example: {'reasoning': example['shorten_reasoning']})
            # 使用大模型，缩短reasoning长度
            print(f"1. shorten reasoning for {subset}")
            if iter != iteration - 1:
                ds = shorten(ds, 32)
            print(f"2. boost answer for {subset}")
            ds = boost(ds, llm)
            print(f"3. verify answer for {subset}")
            right_ds, wrong_ds = verify(ds)
            print(f"Right answer: {len(right_ds)}, Wrong answer: {len(wrong_ds)}")
            print(f"finish for {subset}")
            right_ds.to_json(f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_shorten_iter{iter}_data.jsonl", orient="records", lines=True)

            if len(wrong_ds) > 0:
                all_wrong_ds.append(wrong_ds)
                wrong_ds.to_json(f"/public/data0/NLP/users/wucanhui.volcano/output/{subset}_wrong_iter{iter}_data.jsonl", orient="records", lines=True)
        
        print(f"finish iteration {iter}")

    if all_wrong_ds:
        combined_wrong_ds = concatenate_datasets(all_wrong_ds)
        combined_wrong_ds.to_json(f"/public/data0/NLP/users/wucanhui.volcano/output/all_wrong_data.jsonl", orient="records", lines=True)
        print(f"Saved {len(combined_wrong_ds)} wrong answers to all_wrong_data.jsonl")
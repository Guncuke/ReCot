from datasets import Dataset
from vllm import LLM, SamplingParams
from open_thoughts.prompt import DEEPSEEK_R1_SYSTEM_PROMPT, format_code_prompt
from typing import List


def boost(ds: Dataset, llm: LLM) -> Dataset:
    """
    Boost the answer from prompt and reasoning using batch processing.
    """
    sampling_params = SamplingParams(temperature=0.6, max_tokens=16384)
    
    domain = ds[0]['domain']
    input_texts = []
    
    # 准备所有输入文本
    for item in ds:
        if domain == "puzzle":
            input_text = f"<｜begin▁of▁sentence｜>{DEEPSEEK_R1_SYSTEM_PROMPT}<｜User｜>{item['question']}<｜Assistant｜><think>\n{item['shorten_reasoning']}\n</think>"
        elif domain == "math":
            input_text = f"<｜begin▁of▁sentence｜>{DEEPSEEK_R1_SYSTEM_PROMPT}<｜User｜>Return your final response within \\boxed{{}}. {item['question']}<｜Assistant｜><think>\n{item['shorten_reasoning']}\n</think>"
        else:
            input_text = f"<｜begin▁of▁sentence｜>{DEEPSEEK_R1_SYSTEM_PROMPT}<｜User｜>{format_code_prompt(item)}<｜Assistant｜><think>\n{item['shorten_reasoning']}\n</think>"
        input_texts.append(input_text)
    
    # 批量生成答案
    outputs = llm.generate(input_texts, sampling_params)
    
    # 处理结果
    boosted_answers = [output.outputs[0].text.strip() for output in outputs]
    
    # 更新数据集
    ds = ds.map(lambda example: {"original_deepseek_solution": example.get("deepseek_solution", "")})
    return ds.map(lambda example, idx: {"deepseek_solution": boosted_answers[idx]}, with_indices=True)
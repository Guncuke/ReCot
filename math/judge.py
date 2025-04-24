import os
from openai import OpenAI
from datasets import Dataset
from typing import List, Dict, Any


def create_prompt(solution: str, ground_truth: str) -> str:
    return f"""
    You are a judge that evaluates the correctness of a solution.
    You will be given a proposed solution and a ground truth solution.
    You will need to determine if the proposed solution is correct.
    The proposed solution must arrive at the ground truth solution.

    PROPOSED SOLUTION: {solution}
    GROUND TRUTH SOLUTION: {ground_truth}

    Return your answer directly, True or False.
    If the proposed solution is correct, return True.
    If the proposed solution is incorrect, return False.
    """

def process_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[bool]]:
    client = OpenAI(
        api_key="sk-ad80c93f1b014a8fbc3997dc2e157293", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    results = []
    
    for i in range(len(examples["deepseek_solution"])):
        solution = examples["deepseek_solution"][i]
        ground_truth = examples["solution"][i]
        
        formatted_prompt = create_prompt(solution, ground_truth)
        
        response = client.chat.completions.create(
            model="qwen-plus",
            temperature=0.1,
            top_p=0.8,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            stream=False
        )
        
        response_text = response.choices[0].message.content.strip()
        correct = response_text == "True"
        results.append(correct)
    
    return {"correct": results}

def math_judge(ds: Dataset) -> Dataset:
    
    required_columns = ["solution", "deepseek_solution"]
    for col in required_columns:
        if col not in ds.column_names:
            raise ValueError(f"数据集缺少必要的列: {col}")
    

    return ds.map(
        process_batch,
        batched=True,
        batch_size=8,
        desc="math judge"
    )
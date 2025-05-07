import os
import time
from openai import OpenAI
from datasets import Dataset
from typing import List, Dict, Any



def create_prompt(solution: str, ground_truth: str) -> str:
    return f"""
    You are a judge that evaluates the correctness of a solution.
    You will be given a solution and a ground truth solution.
    You will need to determine if the solution is correct.
    Answers are in the format of \\boxed{{}}.

    SOLUTION: {solution}
    GROUND TRUTH SOLUTION: {ground_truth}

    Return your answer directly, True or False.
    If the proposed solution is correct, return True.
    If the proposed solution is incorrect, return False.
    """

def process_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[bool]]:
    # client = OpenAI(
    #     api_key="sk-c0010e3aa3014e97a9bed2191795480a", 
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    client = OpenAI(
        api_key="JcUSnxQgmlWodneR2Owapxi7MwIwHImX", 
        base_url="http://inferential-api-service-apipre.omniforce.svc.lf06.n.jd.local/v1",
    )
    
    results = []
    
    for i in range(len(examples["deepseek_solution"])):
        solution = examples["deepseek_solution"][i]
        ground_truth = examples["solution"][i]
        formatted_prompt = create_prompt(solution, ground_truth)
        
        max_retries = 3  # 最大重试次数
        retry_delay = 1   # 初始延迟秒数
        correct = False
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="Chatrhino-81B-Pro",
                    temperature=0.1,
                    top_p=0.8,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    stream=False
                )
                response_text = response.choices[0].message.content.strip()
                correct = response_text == "True"
                break
                
            except Exception as e:
                print(f"异常（尝试 {attempt+1}/{max_retries}）: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                continue
        
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
        num_proc=8,
        desc="math judge"
    )
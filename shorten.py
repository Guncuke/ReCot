import time
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from datasets import Dataset


def process_sample(args):
    client, prompt, example, i, max_retries, initial_retry_delay = args

    if 'shorten_reasoning' not in example:
        reasoning = example.get('reasoning', '')
    else:
        reasoning = example.get('shorten_reasoning', '')
    
    formatted_prompt = prompt.format(reasoning=reasoning)
    
    retry_delay = initial_retry_delay
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Chatrhino-81B-Pro",
                temperature=0.1,
                top_p=0.8,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                stream=False
            )
            
            # Extract shortened reasoning
            shorten_reasoning = response.choices[0].message.content.strip()
            return i, shorten_reasoning
            
        except Exception as e:
            print(f"Sample {i} - Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Processing sample {i} failed, keeping original reasoning")
                return i, reasoning


def shorten(ds: Dataset, num_workers: int = 32):
    # client = OpenAI(
    #     api_key="sk-c0010e3aa3014e97a9bed2191795480a", 
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    client = OpenAI(
        api_key="JcUSnxQgmlWodneR2Owapxi7MwIwHImX", 
        base_url="http://inferential-api-service-apipre.omniforce.svc.lf06.n.jd.local/v1",
    )
    
    # Define prompt template for shortening reasoning
    prompt_template = """Please moderately shorten the following reasoning process. Maintain all core logic, key steps, and essential intermediate calculations. Only remove redundant explanations and repetitive checks. The shortened version should preserve about 80% of the original content and lead to the same answer. Directly give the shortened reasoning without any additional explanation or context:
    Reasoning:
    {reasoning}
    Shortened reasoning:"""
    
    max_retries = 3
    initial_retry_delay = 2
    
    result_dict = {
        'shorten_reasoning': [''] * len(ds)
    }
    
    for column in ds.column_names:
        result_dict[column] = ds[column]
    
    args_list = [
        (client, prompt_template, ds[i], i, max_retries, initial_retry_delay)
        for i in range(len(ds))
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sample, args) for args in args_list]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(ds), desc="Shortening reasoning"):
            try:
                i, shorten_reasoning = future.result()
                result_dict['shorten_reasoning'][i] = shorten_reasoning
            except Exception as e:
                print(f"Error processing a sample: {str(e)}")
    
    original_lengths = [len(reasoning) for reasoning in ds['reasoning']]
    shortened_lengths = [len(reasoning) for reasoning in result_dict['shorten_reasoning']]
    original_avg_length = sum(original_lengths) / len(original_lengths)
    shortened_avg_length = sum(shortened_lengths) / len(shortened_lengths)
    print(f"Original average length: {original_avg_length}")
    print(f"Shortened average length: {shortened_avg_length}")

    return Dataset.from_dict(result_dict)

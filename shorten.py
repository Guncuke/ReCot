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
    
    # Construct prompt
    formatted_prompt = prompt.format(reasoning=reasoning)
    
    # Call API with retry mechanism
    retry_delay = initial_retry_delay
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen-plus",
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


def shorten(ds, num_workers=5):
    client = OpenAI(
        api_key="sk-ad80c93f1b014a8fbc3997dc2e157293", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # Define prompt template for shortening reasoning
    prompt_template = """Please moderately shorten the following reasoning process. Maintain all core logic, key steps, and essential intermediate calculations. Only remove redundant explanations and repetitive checks. The shortened version should preserve about 80% of the original content and lead to the same answer. Directly give the shortened result without any additional explanation or context:
    Reasoning:
    {reasoning}
    Shortened reasoning:"""
    
    max_retries = 3
    initial_retry_delay = 2
    
    # 创建临时字典以保存结果
    result_dict = {
        'shorten_reasoning': [''] * len(ds)
    }
    
    # 确保所有其他列都被保留
    for column in ds.column_names:
        result_dict[column] = ds[column]
    
    # Prepare arguments for each sample
    args_list = [
        (client, prompt_template, ds[i], i, max_retries, initial_retry_delay)
        for i in range(len(ds))
    ]
    
    # Process samples using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sample, args) for args in args_list]
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(ds), desc="Shortening reasoning"):
            try:
                i, shorten_reasoning = future.result()
                result_dict['shorten_reasoning'][i] = shorten_reasoning
            except Exception as e:
                print(f"Error processing a sample: {str(e)}")
    
    # 缩短结束，统计缩短前后的平均长度
    original_lengths = [len(reasoning) for reasoning in ds['reasoning']]
    shortened_lengths = [len(reasoning) for reasoning in result_dict['shorten_reasoning']]
    original_avg_length = sum(original_lengths) / len(original_lengths)
    shortened_avg_length = sum(shortened_lengths) / len(shortened_lengths)
    print(f"Original average length: {original_avg_length}")
    print(f"Shortened average length: {shortened_avg_length}")

    # 将结果转换为Dataset对象
    return Dataset.from_dict(result_dict)


if __name__ == "__main__":
    # 使用Dataset创建示例
    from datasets import Dataset
    
    ds = Dataset.from_dict({
        "reasoning": [
            "Okay, so I have this problem where I need to find the number of cases where the product of three natural numbers x, y, z is divisible by 10, given that each of x, y, z is between 1 and 6 inclusive. Hmm, let me think about how to approach this.\n\nFirst, I know that a number is divisible by 10 if it has both 2 and 5 as prime factors. So, for the product xyz to be divisible by 10, at least one of x, y, z must be even (to provide the factor of 2) and at least one of them must be 5 (to provide the factor of 5). \n\nSo, maybe I can approach this by considering the total number of possible triples (x, y, z) and subtracting those that don't meet the criteria. That is, total cases minus the cases where the product isn't divisible by 10. \n\nLet me calculate the total number of possible triples first. Since each of x, y, z can be any number from 1 to 6, there are 6 choices for each. So, the total number of triples is 6 * 6 * 6 = 216.\n\nNow, I need to find how many of these triples don't have a product divisible by 10. These would be the cases where either there's no even number (so no factor of 2) or no 5 in the product. But wait, actually, it's the cases where the product lacks at least one of the necessary factors. So, using the principle of inclusion-exclusion, I can calculate the number of triples missing a factor of 2, the number missing a factor of 5, and then add back in the cases where both are missing because they've been subtracted twice.\n\nLet me define:\n- A: the set of triples where there's no even number (so no factor of 2).\n- B: the set of triples where there's no 5 (so no factor of 5).\n\nThen, the number of triples where the product isn't divisible by 10 is |A \u222a B| = |A| + |B| - |A \u2229 B|.\n\nSo, I need to compute |A|, |B|, and |A \u2229 B|.\n\nStarting with |A|: the number of triples with no even numbers. The numbers from 1 to 6 that are odd are 1, 3, 5. So, each of x, y, z can be 1, 3, or 5. That's 3 choices for each, so |A| = 3 * 3 * 3 = 27.\n\nNext, |B|: the number of triples with no 5s. The numbers from 1 to 6 excluding 5 are 1, 2, 3, 4, 6. That's 5 choices for each of x, y, z. So, |B| = 5 * 5 * 5 = 125.\n\nNow, |A \u2229 B|: the number of triples with no even numbers and no 5s. So, numbers can't be even or 5. The available numbers are 1, 3. That's 2 choices for each of x, y, z. So, |A \u2229 B| = 2 * 2 * 2 = 8.\n\nPutting it all together: |A \u222a B| = 27 + 125 - 8 = 144.\n\nTherefore, the number of triples where the product is divisible by 10 is the total triples minus those not divisible by 10: 216 - 144 = 72.\n\nWait, let me double-check that. So, total triples: 216. Subtract the ones without any 2s (27) and without any 5s (125), but then add back the ones that are missing both (8). So, 27 + 125 - 8 = 144. Then, 216 - 144 = 72. Hmm, that seems right.\n\nAlternatively, maybe I can approach it directly by counting the number of triples that have at least one 5 and at least one even number. Let me see if that gives the same result.\n\nFirst, count the number of triples that include at least one 5. The total number of triples with at least one 5 is total triples minus triples with no 5s: 216 - 125 = 91.\n\nBut among these 91 triples, some might not have any even numbers. So, we need to subtract those. The number of triples with at least one 5 but no even numbers is the number of triples where all numbers are odd and at least one is 5. \n\nWait, so the number of triples with all numbers odd is 3^3 = 27, as before. The number of triples with all numbers odd and no 5s is 2^3 = 8, as before. So, the number of triples with at least one 5 and all numbers odd is 27 - 8 = 19.\n\nTherefore, the number of triples with at least one 5 and at least one even number is 91 - 19 = 72. That matches my previous result. So, that's reassuring.\n\nAlternatively, another way: for the product to be divisible by 10, we need at least one multiple of 2 and at least one multiple of 5. So, we can think of it as the intersection of two events: having at least one even number and having at least one 5.\n\nSo, the number of triples with at least one even number is total triples minus triples with all odd numbers: 216 - 27 = 189.\n\nThe number of triples with at least one 5 is 91 as before.\n\nBut the intersection is the number of triples that have at least one even and at least one 5. Which is what we're trying to find. So, using inclusion-exclusion again, it's |A| + |B| - |A \u222a B|, but wait, no, that's not directly applicable here.\n\nWait, maybe I can use the formula for the intersection: |A \u2229 B| = |A| + |B| - |A \u222a B|. But in this case, A is at least one even, B is at least one 5. So, |A \u2229 B| = |A| + |B| - |A \u222a B|. But |A \u222a B| is the number of triples with at least one even or at least one 5, which is total triples minus triples with neither even nor 5. The triples with neither even nor 5 are the ones with all numbers odd and not 5, which is 2^3 = 8. So, |A \u222a B| = 216 - 8 = 208. Therefore, |A \u2229 B| = 189 + 91 - 208 = 72. Yep, same answer.\n\nSo, all methods point to 72. Hmm, seems consistent.\n\nJust to make sure, let me think of another approach. Maybe count the number of ways to have at least one 5 and at least one even.\n\nFirst, choose positions for the 5s and the even numbers.\n\nBut that might get complicated because of overlapping. Alternatively, maybe fix the number of 5s and evens.\n\nWait, perhaps it's better to think in terms of cases:\n\nCase 1: Exactly one 5, and at least one even.\n\nCase 2: Exactly two 5s, and at least one even.\n\nCase 3: Exactly three 5s, and at least one even.\n\nBut wait, actually, since we need at least one 5 and at least one even, regardless of how many 5s there are (1, 2, or 3), as long as there's at least one even.\n\nSo, let's compute for each number of 5s:\n\nNumber of triples with exactly k 5s and at least one even.\n\nFor k=1: Choose which position is 5: C(3,1)=3. The other two positions must have at least one even. The numbers available for the other two positions are 1,2,3,4,6 (since 5 is already used). Among these, the numbers that are even are 2,4,6, so 3 evens and 2 odds (1,3). So, for each of the two positions, the number of ways to have at least one even is total possibilities minus all odds. Total for two positions: 5^2=25. All odds: 2^2=4. So, 25 - 4 = 21. So, for k=1: 3 * 21 = 63.\n\nFor k=2: Choose which two positions are 5s: C(3,2)=3. The remaining position must be even. The remaining number can be 2,4,6, so 3 choices. So, 3 * 3 = 9.\n\nFor k=3: All three are 5s. But in this case, we need at least one even, but all are 5s, which are odd. So, this case contributes 0.\n\nSo, total number of triples is 63 + 9 + 0 = 72. Yep, same result.\n\nTherefore, I'm confident that the answer is 72.\n\n**Final Answer**\nThe number of cases is \\boxed{72}.\n"
        ]
    })
    
    # 设置并行工作线程数，可以根据需要调整
    num_workers = 50
    shortened_ds = shorten(ds, num_workers)
    
    # 打印结果
    for i in range(len(shortened_ds)):
        print(shortened_ds[i]['shorten_reasoning'])
        print(f"original length {len(shortened_ds[i]['reasoning'])}, shortened length {len(shortened_ds[i]['shorten_reasoning'])}")
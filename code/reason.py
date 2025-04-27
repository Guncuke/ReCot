import os
from bespokelabs import curator
from open_thoughts.prompt import DEEPSEEK_R1_SYSTEM_PROMPT, format_code_prompt


class Reasoner(curator.LLM):
    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        formatted_prompt = format_code_prompt(input)
        return [
            {"role": "system", "content": DEEPSEEK_R1_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt},
        ]
    
    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        input["reasoning"] = response.split('</think>')[0]
        input["deepseek_solution"] = response.split('</think>')[1]
        input["formatted_prompt"] = format_code_prompt(input)
        return input


def reason(ds):
    reasoner = Reasoner( 
        model_name="/public/data0/NLP/users/wucanhui.volcano/models/DeepSeek-R1-Distill-Qwen-7B", 
        backend="vllm",
        generation_params={
            "temperature": 0.1,
        },
        backend_params={ 
            "tensor_parallel_size": 4, # Adjust based on GPU count 
            "gpu_memory_utilization": 0.7,
            "require_all_responses" : False,
            "max_model_length": 16384,
            "max_tokens": 16384,
        }
    )
    return reasoner(ds)


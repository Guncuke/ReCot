import os

from bespokelabs import curator

from open_thoughts import prompt
from open_thoughts.reason import mocked_reasoner


class Reasoner(curator.LLM):
    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        return [
            {"role": "system", "content": prompt.DEEPSEEK_R1_SYSTEM_PROMPT},
            {"role": "user", "content": f"Return your final response within \\boxed{{}}. {input['question']}"},
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        return {
            "question": input["question"],
            "reasoning": response.split('</think>')[0],
            "deepseek_solution": response.split('</think>')[1],
            "solution": input["solution"],
            "domain": input["domain"],
        }


def reason(ds):
    if os.environ.get("MOCK_REASON"):
        return mocked_reasoner(ds, answer_column="solution")
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

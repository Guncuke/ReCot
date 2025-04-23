import os

from bespokelabs import curator
from pydantic import BaseModel


class JudgeResult(BaseModel):
    """Result of the judge's evaluation."""

    correct: bool
    reasoning: str


class MathJudge(curator.LLM):
    """Curator class for processing Numina dataset."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the judge to evaluate the correctness of a solution."""
        return f"""
        You are a judge that evaluates the correctness of a solution.
        You will be given a solution and a ground truth solution.
        You will need to determine if the solution is correct.
        Answers are in the format of \\boxed{{}}.

        SOLUTION: {input["deepseek_solution"]}
        GROUND TRUTH SOLUTION: {input["solution"]}
        Return your answer directly, True or False.
        If the proposed solution is correct, return True.
        If the proposed solution is incorrect, return False.
        """

    def parse(self, input, response):
        """Parse the judge's response to extract correctness and reasoning."""
        return {
            **input,
            "correct": True if response == "True" else False,
        }


math_judge = MathJudge(model_name="/public/data0/NLP/users/tanwentao1/52/project/LLaMA-Factory/hf_models/Qwen/Qwen2.5-72B-Instruct", 
    backend="vllm",
    generation_params={
        "temperature": 0.1,
    },
    backend_params={ 
        "tensor_parallel_size": 8, # Adjust based on GPU count 
        "gpu_memory_utilization": 0.7,
        "require_all_responses" : False,
        "max_model_length": 16384,
        "max_tokens": 16384,
    })

export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_USE_MODELSCOPE=False

cd /public/data0/NLP/users/wucanhui.volcano/lighteval

MODEL_NAME=DeepSeek-R1-Distill-Qwen-7B
MODEL=/public/data0/NLP/users/wucanhui.volcano/models/$MODEL_NAME
MODEL_ARGS="pretrained=$MODEL,dtype=float16,max_model_length=32768,gpu_memory_utilization=0.9,tensor_parallel_size=4,generation_parameters={max_tokens:32768,temperature:0.1,top_p:1.0,stop_token_ids:[151643]}"

OUTPUT_DIR=/public/data0/NLP/users/wucanhui.volcano/lighteval/output/$MODEL

# AIME 2024任务设置
# TASK="custom|aime24|0|0"
TASK="custom|gpqa:diamond|0|0"
export VLLM_WORKER_MULTIPROC_METHOD=spawn && /public/data0/NLP/users/tanwentao1/52/project/Code_o1_project/env/update/lighteval/bin/python -m lighteval vllm $MODEL_ARGS "$TASK" \
    --custom-tasks /public/data0/NLP/users/wucanhui.volcano/lighteval/src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

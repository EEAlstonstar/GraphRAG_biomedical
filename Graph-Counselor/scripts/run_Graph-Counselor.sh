# conda activate graphcounselor

# # is_correct model
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#     --model ../model/Qwen2.5-7B-Instruct \
#     --served-model-name Qwen2.5-7B-Instruct \
#     --tensor-parallel-size 1 \
#     --port 8010 \
#     --host 0.0.0.0 --max_model_len 32768 --gpu-memory-utilization 0.90 > ../scripts/server2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m vllm.entrypoints.openai.api_server \
#     --model ../model/Qwen2.5-72B-Instruct \
#     --served-model-name Qwen2.5-72B-Instruct \
#     --tensor-parallel-size 4 \
#     --port 8020 \
#     --host 0.0.0.0 --gpu-memory-utilization 0.85 > ../scripts/server_test.log 2>&1 &

# sleep 300

# curl -s http://0.0.0.0:8010/v1/completions >/dev/null
# if [ $? -ne 0 ]; then
#     echo "vllm fails, please check the log"
#     exit 1
# fi
# echo "vllm activates"

# curl -s http://0.0.0.0:8020/v1/completions >/dev/null
# if [ $? -ne 0 ]; then
#     echo "vllm fails, please check the log"
#     exit 1
# fi
# echo "vllm activates, start running main script"

# OPENAI_KEY="not real"
# QIANFAN_AK="not real"
# QIANFAN_SK="not real"

# max_steps=10      # max iteration
# max_reflect=2     # max reflection times
# llm_way=vllm      # [vllm, transformer] use vllm or transformer to generate response
# judge_correct=llm #llm groundtruth   # use llm or groundtruth to judge
 
# GPT_version=../model/Qwen2.5-72B-Instruct # Mixtral-8x7B-Instruct-v0.1 Llama-2-13b-chat-hf Mistral-Nemo-Instruct-2407 Qwen2.5-7B-Instruct Meta-Llama-3.1-70B-Instruct Qwen2.5-72B-Instruct
# EVAL_GPT_version=Qwen2.5-7B-Instruct
# SIMPLE_MODEL_NAME=Qwen2.5-72B-Instruct
# REFLECT_version=Qwen2.5-72B-Instruct
# reflexion_strategy=Reflexion #None Last_attempt_and_Reflexion Last_attempt Reflexion
# prompt=multiple              #base short_multiple multiple
# compound_strategy=plan_compound       #None compound plan_compound plan

# for DATASET in dblp #amazon legal biomedical goodreads dblp maple
# do
#     if [ "$DATASET" != "maple" ]; then
#         DATA_PATH=../../data/processed_data/$DATASET
#         SAVE_FILE=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-$DATASET/results.jsonl
#         SAVE_FILE_NOREFLECT=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-$DATASET/result_noreflect.jsonl

#         python ../code/run.py --dataset $DATASET \
#                     --path $DATA_PATH \
#                     --save_file $SAVE_FILE \
#                     --save_file_first $SAVE_FILE_NOREFLECT\
#                     --llm_version $GPT_version \
#                     --reflect_version $REFLECT_version \
#                     --openai_api_key $OPENAI_KEY \
#                     --qianfan_ak $QIANFAN_AK \
#                     --qianfan_sk $QIANFAN_SK \
#                     --max_steps $max_steps \
#                     --reflexion_strategy $reflexion_strategy\
#                     --max_reflect $max_reflect\
#                     --llm_way $llm_way\
#                     --compound_strategy $compound_strategy\
#                     --eval_llm_version $EVAL_GPT_version\
#                     --api_url http://0.0.0.0:8020/v1/completions\
#                     --api_url2 http://0.0.0.0:8010/v1/completions\
#                     --judge_correct $judge_correct\
#                     --reflect_prompt $prompt\
#                     --api_url3 http://0.0.0.0:8020/v1/completions > ../scripts/Qwen2.5-72B-Instruct-$DATASET-plan_reflect_llm7b_multiple.log
#         curl -s http://0.0.0.0:8010/v1/completions > /dev/null
#         if [ $? -ne 0 ]; then
#             echo "vllm fails, please check the log"
#             exit 1
#         fi
#         echo "vllm activates"

#         curl -s http://0.0.0.0:8020/v1/completions > /dev/null
#         if [ $? -ne 0 ]; then
#             echo "vllm fails, please check the log"
#             exit 1
#         fi
#         echo "vllm activates, start running main script"

#     else
#         for SUBDATASET in Biology Chemistry Materials_Science Medicine Physics #Biology Chemistry Materials_Science Medicine Physics
#         do
#             DATA_PATH=../../data/processed_data/maple/$SUBDATASET
#             SAVE_FILE=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-maple-$SUBDATASET/results.jsonl
#             SAVE_FILE_NOREFLECT=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-maple-$SUBDATASET/result_noreflect.jsonl

#             python ../code/run.py --dataset $DATASET \
#                     --path $DATA_PATH \
#                     --save_file $SAVE_FILE \
#                     --save_file_first $SAVE_FILE_NOREFLECT\
#                     --llm_version $GPT_version \
#                     --reflect_version $REFLECT_version \
#                     --openai_api_key $OPENAI_KEY \
#                     --qianfan_ak $QIANFAN_AK \
#                     --qianfan_sk $QIANFAN_SK \
#                     --max_steps $max_steps \
#                     --reflexion_strategy $reflexion_strategy\
#                     --max_reflect $max_reflect\
#                     --llm_way $llm_way\
#                     --compound_strategy $compound_strategy\
#                     --eval_llm_version $EVAL_GPT_version\
#                     --api_url http://0.0.0.0:8020/v1/completions\
#                     --api_url2 http://0.0.0.0:8010/v1/completions\
#                     --judge_correct $judge_correct\
#                     --reflect_prompt $prompt\
#                     --api_url3 http://0.0.0.0:8020/v1/completions > ../scripts/Qwen2.5-72B-Instruct-$DATASET-$maple-plan_reflect_llm7b_multiple.log

#         done
#     fi
# done

#!/bin/bash
# === 4090 终极全自动版 (Auto-Fix & Local Model) ===

# 0. 检查运行目录
if [ ! -d "code" ]; then
    echo "错误：请在 'Graph-Counselor/Graph-Counselor' 根目录下运行！"
    exit 1
fi

# === 配置区：请确认你的本地模型路径 ===
LOCAL_MODEL_PATH="/root/Graph-Counselor/Graph-Counselor/models/qwen/Qwen2.5-7B-Instruct"

# 1. 环境准备
MKL_PATH=$CONDA_PREFIX/lib
PY_CMD="env LD_LIBRARY_PATH=$MKL_PATH:$LD_LIBRARY_PATH python"

echo "===  0. 正在清理旧环境... ==="
# 强力清理旧进程，防止端口冲突
pkill -f vllm
# 【关键】给系统 10 秒钟去回收端口，防止 Address already in use
echo "等待端口释放 (10秒)..."
sleep 10

echo "=== 1. 正在启动 vLLM 服务 (本地模型) ==="
# 启动服务
CUDA_VISIBLE_DEVICES=0 $PY_CMD -m vllm.entrypoints.openai.api_server \
    --model $LOCAL_MODEL_PATH \
    --served-model-name Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --port 8010 \
    --host 0.0.0.0 \
    --max_model_len 16384 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --trust-remote-code > scripts/server_7b.log 2>&1 &

echo "服务启动中... (你可以另开窗口运行 'tail -f scripts/server_7b.log' 看日志)"

# === 智能等待循环 ===
# 每 5 秒检查一次，直到服务通了才往下跑
echo "正在等待服务就绪..."
while ! curl -s http://0.0.0.0:8010/v1/models > /dev/null; do
    sleep 5
    echo -n "." 
done
echo ""
echo "vLLM 服务已就绪！开始运行主程序..."

# 2. 运行主程序
OPENAI_KEY="empty"
MODEL_NAME=Qwen2.5-7B-Instruct

for DATASET in biomedical
do
    echo "=== 2. 开始处理数据集: $DATASET ==="
    DATA_PATH=../data/processed_data/$DATASET
    SAVE_FILE=../results/$MODEL_NAME/${DATASET}_test.jsonl
    
    python -u code/run.py --dataset $DATASET \
        --path $DATA_PATH \
        --save_file $SAVE_FILE \
        --save_file_first ${SAVE_FILE}_noreflect \
        --llm_version $MODEL_NAME \
        --reflect_version $MODEL_NAME \
        --eval_llm_version $MODEL_NAME \
        --openai_api_key $OPENAI_KEY \
        --max_steps 5 \
        --reflexion_strategy Reflexion \
        --max_reflect 2 \
        --llm_way vllm \
        --compound_strategy plan_compound \
        --api_url http://0.0.0.0:8010/v1/completions \
        --api_url2 http://0.0.0.0:8010/v1/completions \
        --api_url3 http://0.0.0.0:8010/v1/completions \
        --judge_correct llm \
        --use_dynamic_fewshot True \
        --few_shot_bank_file "few_shot_bank.json" \
        --dynamic_k 3 \
        --reflect_prompt multiple > scripts/Run-4090-$DATASET.log 2>&1
    
    echo "数据集 $DATASET 跑完了！日志: scripts/Run-4090-$DATASET.log"
done

# 跑完自动清理
echo "=== 全部完成，正在关闭服务... ==="
pkill -f vllm
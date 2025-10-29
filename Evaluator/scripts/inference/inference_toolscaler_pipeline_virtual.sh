export TOOLBENCH_KEY="your key"
export OPENAI_KEY="your key"
export PYTHONPATH=./
export SERVICE_URL="http://localhost:8080/virtual"
export CUDA_VISIBLE_DEVICES=3

model_path="llama-3-8b-full/sft/gist_dfs_2@2"
indexing="hierarchical"
template="llama-3"
model_name="gist2@2"

export OUTPUT_DIR="data/answer/${model_name}/"
stage="G3"
group="instruction"

if [ $indexing == "Atomic" ]; then
    backbone_model="toolgen_atomic"
else
    backbone_model="toolgen"
fi

mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/${stage}_${group}
CUDA_VISIBLE_DEVICES=3 python evaluation/toolbench/inference/qa_pipeline_multithread.py \
    --chatgpt_model gpt-4o-mini-2024-07-18 \
    --base_url your base url \
    --model_path ${model_path} \
    --template ${template} \
    --indexing ${indexing} \
    --tool_root_dir data/toolenv/tools \
    --backbone_model ${backbone_model} \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method CoT@1 \
    --input_query_file data/solvable_queries/test_instruction/${stage}_${group}.json \
    --output_answer_file $OUTPUT_DIR/${stage}_${group} \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 1 \
    --function_provider all
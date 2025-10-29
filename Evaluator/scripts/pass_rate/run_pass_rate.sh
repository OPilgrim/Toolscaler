export OPENAI_KEY="your key"
export API_POOL_FILE=openai_keys.json
export CONVERTED_ANSWER_PATH=data/model_predictions_converted
export SAVE_PATH=data/results/pass_rate
mkdir -p ${SAVE_PATH}
export CANDIDATE_MODEL=gist2@2
TEST_SET="G3_instruction"
export EVAL_MODEL=gpt-4o-mini-2024-07-18
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}


python -m evaluation.toolbench.tooleval.eval_pass_rate \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids data/solvable_queries/test_query_ids \
    --max_eval_threads 3 \
    --evaluate_times 3 \
    --test_set ${TEST_SET}
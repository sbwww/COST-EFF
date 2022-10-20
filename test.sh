export CUDA_VISIBLE_DEVICES=0

N_GPU=1                 # currently not well-adapted to multi-gpu training

SEQ_LEN=128             # 128 input tokens following BERT
EVAL_BATCH=32           # 32 / 64 evaluating batch size
EE_EVAL_BATCH=1         # 1 batch size for dynamic early exiting

# pooler setting
HIDDEN_MODE="concat"    # concat / avg last N hidden states
LAST_N_HIDDEN=1         # N = 1
POOLING_MODE="cls"      # use cls / avg of tokens

POOLER_SETTING=${HIDDEN_MODE}_${LAST_N_HIDDEN}_${POOLING_MODE}

# Compression settings
# number of layers, not changed
NUM_OF_LAYERS_TO_KEEP=12

# number of attention heads, integer in [1, 12]
NUM_OF_ATTN_HEADS_TO_KEEP=6

# intermidiate size of FFN, two scaling approach both work
HIDDEN_DIM_OF_FFN=$((3072 / 12 * ${NUM_OF_ATTN_HEADS_TO_KEEP}))
# HIDDEN_DIM_OF_FFN=$((3072 / 12 * ${NUM_OF_ATTN_HEADS_TO_KEEP} / 2))

# hidden size in word embedding after SVD
MATRIX_RANK_OF_EMB_FACTORIZATION=$((768 / 12 * ${NUM_OF_ATTN_HEADS_TO_KEEP}))

MODEL_CONF_NAME=a${NUM_OF_ATTN_HEADS_TO_KEEP}_l${NUM_OF_LAYERS_TO_KEEP}_f${HIDDEN_DIM_OF_FFN}_e${MATRIX_RANK_OF_EMB_FACTORIZATION}


# evaluating finetuned model on dev set
Eval_Finetune() {
    python -m Costeff.run_glue_costeff \
        --student_model ./models/finetuned_model/${TASK_NAME} \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/finetuned_model/${TASK_NAME} \
        --do_eval \
        --max_seq_length ${SEQ_LEN} \
        --eval_batch_size ${EE_EVAL_BATCH} \
        --do_lower_case \
        --output_hidden_states \
        --output_attentions \
        --overwrite_output_dir
}

# evaluating and profiling finetuned model forward FLOPs on dev set
# FLOPs of finetuned model is used to compute FLOPs reduction
Profile_finetune() {
    python -m Costeff.run_glue_costeff \
        --student_model ./models/finetuned_model/${TASK_NAME} \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/finetuned_model/${TASK_NAME} \
        --do_eval \
        --do_prof \
        --train_ft \
        --max_seq_length ${SEQ_LEN} \
        --eval_batch_size ${EE_EVAL_BATCH} \
        --do_lower_case \
        --output_hidden_states \
        --max_profile_samples 2000 \
        --overwrite_output_dir
}

# evaluating COST-EFF model on dev set
Eval_costeff() {
    python -m Costeff.run_glue_costeff \
        --student_model ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --do_eval \
        --max_seq_length ${SEQ_LEN} \
        --eval_batch_size ${EE_EVAL_BATCH} \
        --do_lower_case \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --highway_mode \
        --early_exit_entropy $1 \
        --overwrite_output_dir
}

# evaluating and profiling COST-EFF forward FLOPs on dev set
Profile_costeff() {
    python -m Costeff.run_glue_costeff \
        --student_model ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --do_eval \
        --do_prof \
        --max_seq_length ${SEQ_LEN} \
        --eval_batch_size ${EE_EVAL_BATCH} \
        --do_lower_case \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --max_profile_samples 2000 \
        --highway_mode \
        --early_exit_entropy $1 \
        --overwrite_output_dir
}

# evaluating each layer performance of TA model on dev set
Eval_layer_ta() {
    python -m Costeff.run_glue_costeff \
        --student_model ./models/TA_model/${TASK_NAME}/${POOLER_SETTING}/1t \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/TA_model/${TASK_NAME}/${POOLER_SETTING}/1t \
        --do_eval \
        --eval_layer \
        --max_seq_length ${SEQ_LEN} \
        --eval_batch_size ${EVAL_BATCH} \
        --do_lower_case \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --highway_mode \
        --overwrite_output_dir
}

# evaluating each layer performance of COST-EFF model on dev set
Eval_layer_costeff() {
    python -m Costeff.run_glue_costeff \
        --student_model ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --do_eval \
        --eval_layer \
        --max_seq_length ${SEQ_LEN} \
        --eval_batch_size ${EVAL_BATCH} \
        --do_lower_case \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --highway_mode \
        --overwrite_output_dir
}

# evaluating each layer performance of COST-EFF model on test set
Test_costeff() {
    python -m Costeff.run_glue_costeff \
        --student_model ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --do_test \
        --max_seq_length ${SEQ_LEN} \
        --eval_batch_size ${EE_EVAL_BATCH} \
        --do_lower_case \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --max_profile_samples 2000 \
        --highway_mode \
        --early_exit_entropy $1 \
        --overwrite_output_dir
}

TASKS=("MRPC" "SST-2" "QNLI" "MNLI" "CoLA" "QQP")
ENTROPIES=("0.6" "0.55" "0.5" "0.4" "0.3" "0.2" "0.1" "0.05" "0.0")
# MNLI has 3 labels, so the entropy upperbound is higher
# ENTROPIES=("1.0" "0.9" "0.8" "0.6" "0.4" "0.2" "0.0")

for ((i = 0; i < 1; i++)); do
    TASK_NAME=${TASKS[i]}
    Profile_finetune
    for ((j = 0; j < 9; j++)); do
        Profile_costeff ${ENTROPIES[j]}
    done
    Eval_layer_ta
    Eval_layer_costeff
done

export CUDA_VISIBLE_DEVICES=0

N_GPU=1                 # currently not well-adapted to multi-gpu training

SEQ_LEN=128             # 128 input tokens following BERT
TRAIN_BATCH=32          # 32 / 64 training batch size
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

# ordinary finetuning
Finetune() {
    python -m Costeff.run_glue_costeff \
        --student_model ./models/pretrained_model/bert-base-uncased \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/finetuned_model/${TASK_NAME} \
        --max_seq_length ${SEQ_LEN} \
        --num_train_epochs ${EPOCH} \
        --train_batch_size ${TRAIN_BATCH} \
        --eval_batch_size ${EVAL_BATCH} \
        --eval_step ${EVAL_STEP} \
        --lr_schedule warmup_linear \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --learning_rate ${LR} \
        --overwrite_output_dir \
        --train_ft \
        --do_lower_case
}

# training TA model
Train_TA() {
    # start with representation-only distillation for a small epochs
    python -m Costeff.run_glue_costeff \
        --teacher_model ./models/finetuned_model/${TASK_NAME} \
        --student_model ./models/pretrained_model/bert-base-uncased \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/TA_model/${TASK_NAME}/${POOLER_SETTING}/1t \
        --max_seq_length ${SEQ_LEN} \
        --num_train_epochs $((${EPOCH} / 3)) \
        --train_batch_size ${TRAIN_BATCH} \
        --eval_batch_size ${EVAL_BATCH} \
        --eval_step ${EVAL_STEP} \
        --lr_schedule warmup_linear \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --learning_rate ${LR} \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --repr_distill \
        --overwrite_output_dir \
        --train_ta \
        --train_1t \
        --highway_mode \
        --exit_start ${EXIT_START} \
        --do_lower_case

    # prediction and representation distillation
    python -m Costeff.run_glue_costeff \
        --teacher_model ./models/finetuned_model/${TASK_NAME} \
        --student_model ./models/TA_model/${TASK_NAME}/${POOLER_SETTING}/1t \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/TA_model/${TASK_NAME}/${POOLER_SETTING}/1t \
        --max_seq_length ${SEQ_LEN} \
        --num_train_epochs ${EPOCH} \
        --train_batch_size ${TRAIN_BATCH} \
        --eval_batch_size ${EVAL_BATCH} \
        --eval_step ${EVAL_STEP} \
        --lr_schedule warmup_linear \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --learning_rate ${LR} \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --pred_distill \
        --overwrite_output_dir \
        --train_ta \
        --train_1t \
        --highway_mode \
        --exit_start ${EXIT_START} \
        --do_lower_case
}

# iterative pruning (i.e., train (repr-only, pred & repr) -> prune -> train -> prune -> ... -> stop)
Prune_costeff() {
    python -m Costeff.run_glue_costeff \
        --teacher_model ./models/TA_model/${TASK_NAME}/${POOLER_SETTING}/1t \
        --student_model ./models/TA_model/${TASK_NAME}/${POOLER_SETTING}/1t \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --num_train_epochs ${EPOCH} \
        --max_seq_length ${SEQ_LEN} \
        --train_batch_size ${TRAIN_BATCH} \
        --eval_batch_size ${EVAL_BATCH} \
        --eval_step ${EVAL_STEP} \
        --lr_schedule warmup_linear \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --learning_rate ${LR} \
        --lr_restore_factor ${LR_RESTORE} \
        --prun_period_proportion ${PRUNE_RATE} \
        --keep_heads ${NUM_OF_ATTN_HEADS_TO_KEEP} \
        --keep_layers ${NUM_OF_LAYERS_TO_KEEP} \
        --emb_hidden_dim ${MATRIX_RANK_OF_EMB_FACTORIZATION} \
        --ffn_hidden_dim ${HIDDEN_DIM_OF_FFN} \
        --depth_or_width width \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --repr_distill \
        --pred_distill \
        --overwrite_output_dir \
        --train_costeff \
        --train_1t \
        --highway_mode \
        --sep_taylor \
        --internal_loss \
        --taylor_proportion 1.0 \
        --exit_start ${EXIT_START} \
        --repr_proportion ${REPR_PROP} \
        --do_lower_case
}

# representation-only distillation
Train_costeff_repr() {
    python -m Costeff.run_glue_costeff \
        --teacher_model ./models/TA_model/${TASK_NAME}/${POOLER_SETTING}/1t \
        --student_model ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --num_train_epochs ${EPOCH} \
        --max_seq_length ${SEQ_LEN} \
        --train_batch_size ${TRAIN_BATCH} \
        --eval_batch_size ${EVAL_BATCH} \
        --eval_step ${EVAL_STEP} \
        --lr_schedule warmup_linear \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --learning_rate ${LR} \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --repr_distill \
        --overwrite_output_dir \
        --train_costeff \
        --train_1t \
        --exit_start ${EXIT_START} \
        --highway_mode \
        --do_lower_case
}

# prediction and representation distillation
Train_costeff_pred_and_repr() {
    python -m Costeff.run_glue_costeff \
        --teacher_model ./models/TA_model/${TASK_NAME}/${POOLER_SETTING}/1t \
        --student_model ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --task_name ${TASK_NAME} \
        --data_dir ./data/glue/${TASK_NAME} \
        --output_dir ./models/costeff_model/${TASK_NAME}/${MODEL_CONF_NAME}/${POOLER_SETTING}/1t \
        --num_train_epochs ${EPOCH} \
        --max_seq_length ${SEQ_LEN} \
        --train_batch_size ${TRAIN_BATCH} \
        --eval_batch_size ${EVAL_BATCH} \
        --lr_schedule warmup_linear \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --learning_rate ${LR} \
        --output_hidden_states \
        --hidden_mode ${HIDDEN_MODE} \
        --pooling_mode ${POOLING_MODE} \
        --last_n_hidden ${LAST_N_HIDDEN} \
        --repr_distill \
        --pred_distill \
        --eval_step ${EVAL_STEP} \
        --overwrite_output_dir \
        --train_costeff \
        --train_1t \
        --exit_start ${EXIT_START} \
        --highway_mode \
        --do_lower_case
}

TASKs=("MRPC" "SST-2" "QNLI" "MNLI" "CoLA" "QQP")
EVAL_STEPs=(50 500 1000 2500 100 5000)      # how many training steps between evaluations

EPOCHs_T=(12 5 5 3 8 3)                     # finetuning and TA training epochs

# some empirically set epochs, may not be optimal
case ${NUM_OF_ATTN_HEADS_TO_KEEP} in
2)
    EPOCHs_P=(20 5 5 3 20 3)                # iterative pruning epochs
    PRUNE_RATEs=(2.0 0.5 0.5 0.3 2.0 0.3)   # propotion of pruning per iteration
    EPOCHs_RD=(20 10 8 5 20 5)              # repr-only distill epochs
    EPOCHs_PD=(10 6 5 3 10 3)               # pred distill epochs
    ;;
3)
    EPOCHs_P=(18 5 5 3 18 3)
    PRUNE_RATEs=(2.0 0.55 0.55 0.33 2.0 0.33)
    EPOCHs_RD=(18 10 8 5 18 5)
    EPOCHs_PD=(8 5 4 2 8 2)
    ;;
4)
    EPOCHs_P=(16 4 4 3 16 3)
    PRUNE_RATEs=(2.0 0.5 0.5 0.37 2.0 0.37)
    EPOCHs_RD=(16 9 7 4 16 4)
    EPOCHs_PD=(7 4 4 2 7 2)
    ;;
5)
    EPOCHs_P=(14 4 4 2 14 2)
    PRUNE_RATEs=(2.0 0.55 0.55 0.28 2.0 0.28)
    EPOCHs_RD=(14 9 7 4 14 4)
    EPOCHs_PD=(6 4 3 2 6 2)
    ;;
6)
    EPOCHs_P=(12 3 3 2 12 2)
    PRUNE_RATEs=(2.0 0.5 0.5 0.33 2.0 0.33)
    EPOCHs_RD=(12 8 6 3 10 3)
    EPOCHs_PD=(5 3 3 2 5 2)
    ;;
7)
    EPOCHs_P=(10 2 2 1 10 1)
    PRUNE_RATEs=(2.0 0.4 0.4 0.2 2.0 0.2)
    EPOCHs_RD=(10 7 6 3 10 3)
    EPOCHs_PD=(4 3 2 1 4 1)
    ;;
8)
    EPOCHs_P=(8 2 2 1 8 1)
    PRUNE_RATEs=(2.0 0.5 0.5 0.25 2.0 0.25)
    EPOCHs_RD=(8 6 5 2 8 2)
    EPOCHs_PD=(3 3 2 1 3 1)
    ;;
esac

REPR_PROPs=(0.5 0.7 0.7 0.7 0.5 0.7) # repr-only distill propotion in iterative pruning

EXITS=(1 1 1 1 1 1) # early exit starting layer. For MRPC, QNLI, MNLI, we can start from the 3-rd layer

for ((i = 0; i < 1; i++)); do
    TASK_NAME=${TASKs[i]}

    EVAL_STEP=${EVAL_STEPs[i]}
    EXIT_START=${EXITS[i]}

    EPOCH=${EPOCHs_T[i]}
    # a larger batch size is likely to need a larger learning rate, may not be optimal
    if [ ${TRAIN_BATCH} == 64 ]; then
        LR=3e-5
    elif [ ${TRAIN_BATCH} == 32 ]; then
        LR=2e-5
    fi
    Finetune
    Train_TA

    EPOCH=${EPOCHs_P[i]}
    PRUNE_RATE=${PRUNE_RATEs[i]}
    REPR_PROP=${REPR_PROPs[i]}
    if [ ${TRAIN_BATCH} == 64 ]; then
        LR=1e-5
        LR_RESTORE=0.15 # we enlarge the learning rate during iterative pruning
    elif [ ${TRAIN_BATCH} == 32 ]; then
        LR=1e-5
        LR_RESTORE=0.12
    fi
    Prune_costeff

    EPOCH=${EPOCHs_RD[i]}
    if [ ${TRAIN_BATCH} == 64 ]; then
        LR=4e-5
    elif [ ${TRAIN_BATCH} == 32 ]; then
        LR=3e-5
    fi
    Train_costeff_repr

    EPOCH=${EPOCHs_PD[i]}
    if [ ${TRAIN_BATCH} == 64 ]; then
        LR=3e-5
    elif [ ${TRAIN_BATCH} == 32 ]; then
        LR=2e-5
    fi
    Train_costeff_pred_and_repr

    echo "========== End of Run =========="
done

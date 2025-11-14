model_name=AutoTimes_Llama

# --- Common args ---
ROOT=./dataset/ETT-small/
DATA_PATH=ETTh1.csv
DATA=ETTh1
MODEL_ID=ETTh1_672_96

SEQ=672
LABEL=576
TOKEN=96
BATCH=256
LR=0.0005

# Folder where arrays/plots will be saved
TEST_DIR=long_term_forecast_ETTh1_672_96_AutoTimes_Llama_ETTh1_sl672_ll576_tl96_lr0.0005_bt256_wd0_hd256_hl0_cosTrue_mixTrue_test_0

# Horizons to evaluate
HORIZONS=(96 192 336 720)

echo "== Test-only ETTh1 =="
for H in "${HORIZONS[@]}"; do
  echo ">>> Horizon ${H}"
  "$PY" -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path "$ROOT" \
    --data_path "$DATA_PATH" \
    --model_id "$MODEL_ID" \
    --model "$model_name" \
    --data "$DATA" \
    --seq_len $SEQ \
    --label_len $LABEL \
    --token_len $TOKEN \
    --test_seq_len $SEQ \
    --test_label_len $LABEL \
    --test_pred_len $H \
    --batch_size $BATCH \
    --learning_rate $LR \
    --mlp_hidden_layers 0 \
    --use_amp \
    --gpu 0 \
    --cosine \
    --tmax 10 \
    --mix_embeds \
    --drop_last \
    --save_arrays \
    --test_dir "$TEST_DIR"
    # add --visualize if you also want PNG plots saved
done
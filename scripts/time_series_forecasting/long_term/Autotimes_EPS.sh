#!/usr/bin/env bash

model_name=AutoTimes_Llama

# -----------------------------
# Common EPS config
# -----------------------------
root_path=/largessd/home/muntasir/Desktop/AutoTimes/dataset/panel/univariate
data_path=df_10.csv

# temporal split for EPS panel
train_end_date=2014-12-31
val_end_date=2017-12-31

# sequence / horizon
seq_len=32
label_len=4
token_len=4      # segment length for AR loop
test_seq_len=32
test_label_len=4
test_pred_len=4  # 4 future quarters

batch_size=64
learning_rate=0.0005
train_epochs=2

model_id=EPS_${seq_len}_${test_pred_len}

# This string must match the "setting" format in run.py for test_dir
setting_name=long_term_forecast_${model_id}_${model_name}_EPS_sl${seq_len}_ll${label_len}_tl${token_len}_lr${learning_rate}_bt${batch_size}_wd0_hd256_hl0_cosTrue_mixTrue_test_0

# -----------------------------
# 1) Train EPS model
# -----------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${root_path} \
  --data_path ${data_path} \
  --model_id ${model_id} \
  --model ${model_name} \
  --data EPS \
  --seq_len ${seq_len} \
  --label_len ${label_len} \
  --token_len ${token_len} \
  --test_seq_len ${test_seq_len} \
  --test_label_len ${test_label_len} \
  --test_pred_len ${test_pred_len} \
  --batch_size ${batch_size} \
  --learning_rate ${learning_rate} \
  --mlp_hidden_layers 0 \
  --train_epochs ${train_epochs} \
  --train_end_date ${train_end_date} \
  --val_end_date ${val_end_date} \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --drop_last
#  --mix_embeds \


# -----------------------------
# 2) Test EPS model (save panel CSV)
# -----------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ${root_path} \
  --data_path ${data_path} \
  --model_id ${model_id} \
  --model ${model_name} \
  --data EPS \
  --seq_len ${seq_len} \
  --label_len ${label_len} \
  --token_len ${token_len} \
  --test_seq_len ${test_seq_len} \
  --test_label_len ${test_label_len} \
  --test_pred_len ${test_pred_len} \
  --batch_size ${batch_size} \
  --learning_rate ${learning_rate} \
  --mlp_hidden_layers 0 \
  --train_epochs ${train_epochs} \
  --train_end_date ${train_end_date} \
  --val_end_date ${val_end_date} \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --save_arrays \
  --drop_last \
  --test_dir ${setting_name} \
  --test_file_name checkpoint.pth
#  --mix_embeds \

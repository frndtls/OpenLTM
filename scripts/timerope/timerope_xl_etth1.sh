export CUDA_VISIBLE_DEVICES=0
model_name=timerope
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]
# training one model with a context length
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --duration 3600 \
  --anchor_list 1 60 $((60*60)) $((60*60*24)) $((60*60*24*7)) $((60*60*24*30)) $((60*60*24*365)) \
  --clock_list 12 12 10 10 10 10


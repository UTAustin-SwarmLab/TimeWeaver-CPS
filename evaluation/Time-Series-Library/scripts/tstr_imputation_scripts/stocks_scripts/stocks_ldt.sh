model_name=TimesNet
real_data_dir=''
synthetic_data_dir=''


python -u run.py \
  --task_name tstr_imputation \
  --log_exp_name tstr_imputation_stocks_ldt \
  --synthetic_data_dir $synthetic_data_dir \
  --real_data_dir $real_data_dir \
  --is_training 1 \
  --root_path ./dataset/imputation/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.75_final \
  --mask_rate 0.75 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --batch_size 16 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 3 \
  --top_k 3 \
  --learning_rate 0.001
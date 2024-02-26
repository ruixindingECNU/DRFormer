export CUDA_VISIBLE_DEVICES=1

model_name=DRFormer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 3 \
  --patch_len 16 \
  --stride 4 \
  --sequence_num 3\
  --update_frequency 0.3 \
  --learnable_mask_epoches 0.5 \
  --sparsity 0.5 \
  --death_mode 'magnitude' \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 256

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 3 \
  --patch_len 16 \
  --stride 4 \
  --sequence_num 3\
  --update_frequency 0.3 \
  --learnable_mask_epoches 0.5 \
  --sparsity 0.5 \
  --death_mode 'magnitude' \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 256

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 3 \
  --patch_len 16 \
  --stride 4 \
  --sequence_num 3\
  --update_frequency 0.3 \
  --learnable_mask_epoches 0.5 \
  --sparsity 0.5 \
  --death_mode 'magnitude' \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 256

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 3 \
  --patch_len 16 \
  --stride 4 \
  --sequence_num 3\
  --update_frequency 0.3 \
  --learnable_mask_epoches 0.5 \
  --sparsity 0.5 \
  --death_mode 'magnitude' \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 256
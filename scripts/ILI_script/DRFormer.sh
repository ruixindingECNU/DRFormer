export CUDA_VISIBLE_DEVICES=0

model_name=DRFormer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1 \
  --patch_len 24 \
  --stride 2 \
  --sequence_num 3\
  --update_frequency 0.2 \
  --learnable_mask_epoches 1 \
  --train_epochs 50\
  --lradj 'constant'\
  --patience 50\
  --sparsity 0.3 \
  --death_mode 'magnitude' \
  --learning_rate 0.0001

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1 \
  --patch_len 24 \
  --stride 2 \
  --sequence_num 3\
  --update_frequency 0.2 \
  --learnable_mask_epoches 1 \
  --train_epochs 50\
  --lradj 'constant'\
  --patience 50\
  --sparsity 0.3 \
  --death_mode 'magnitude' \
  --learning_rate 0.0001

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1 \
  --patch_len 24 \
  --stride 2 \
  --sequence_num 3\
  --update_frequency 0.2 \
  --learnable_mask_epoches 1 \
  --train_epochs 50\
  --lradj 'constant'\
  --patience 50\
  --sparsity 0.3 \
  --death_mode 'magnitude' \
  --learning_rate 0.0001

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1 \
  --patch_len 24 \
  --stride 2 \
  --sequence_num 3\
  --update_frequency 0.2 \
  --learnable_mask_epoches 1 \
  --train_epochs 50\
  --lradj 'constant'\
  --patience 50\
  --sparsity 0.3 \
  --death_mode 'magnitude' \
  --learning_rate 0.0001
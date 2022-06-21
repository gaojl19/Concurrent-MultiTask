# python starter/separate_mlp_baseline.py \
#     --n_iter 50 \
#     --eval_interval 1 \
#     --learning_rate 1e-4 \
#     --ep_len 800 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/concurrent.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --gradient_steps 100 \
#     --expert_num 5 \
#     --task_types push-1\
#     --no_cuda 

# python starter/mlp_baseline.py \
#     --n_iter 50 \
#     --eval_interval 1 \
#     --learning_rate 1e-4 \
#     --ep_len 800 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/concurrent.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --gradient_steps 100 \
#     --expert_num 5 \
#     --no_cuda 


# python starter/EM.py \
#     --n_iter 20 \
#     --eval_interval 1 \
#     --learning_rate 1e-4 \
#     --ep_len 300 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/concurrent.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --gradient_steps 100 \
#     --expert_num 5 \
#     --no_cuda \
#     --train_full_data true 


# python starter/MultiNet_baseline.py \
#     --n_iter 20 \
#     --eval_interval 1 \
#     --learning_rate 1e-4 \
#     --ep_len 300 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/concurrent.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --gradient_steps 100 \
#     --expert_num 1 \
#     --no_cuda 

# python starter/VAE_baseline.py \
#     --n_iter 20 \
#     --eval_interval 1 \
#     --learning_rate 1e-4 \
#     --ep_len 400 \
#     --batch_size 64 \
#     --train_batch_size 1 \
#     --config config/concurrent.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --gradient_steps 100 \
#     --expert_num 1 \
#     --no_cuda \
#     --soft true \
#     --train_full_data true 


# python starter/gaussian_baseline.py \
#     --n_iter 2 \
#     --eval_interval 1 \
#     --learning_rate 1e-4 \
#     --ep_len 800 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/concurrent.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --gradient_steps 100 \
#     --expert_num 1 \
#     --no_cuda \
#     --train_full_data true


# python starter/separate_gaussian_baseline.py \
#     --n_iter 20 \
#     --eval_interval 1 \
#     --learning_rate 1e-4 \
#     --ep_len 300 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/concurrent.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --gradient_steps 100 \
#     --expert_num 10 \
#     --task_types push-1\
#     --no_cuda \
#     --train_full_data true
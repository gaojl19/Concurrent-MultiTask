# python starter/gaussian_baseline.py \
#     --n_iter 50 \
#     --eval_interval 1 \
#     --learning_rate 1e-4 \
#     --ep_len 400 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/concurrent.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --gradient_steps 100 \
#     --expert_num 3 \
#     --no_cuda \
#     --test true \
#     --load_action_path ./fig/random_both.json \
#     --load_from_checkpoint ./fig/gaussian_baseline/random_both_success_model.pth


python starter/VAE_baseline.py \
    --n_iter 50 \
    --eval_interval 1 \
    --learning_rate 1e-4 \
    --ep_len 300 \
    --batch_size 64 \
    --train_batch_size 32 \
    --config config/concurrent.json \
    --id MT50_Single_Task \
    --seed 127 \
    --worker_nums 1 \
    --eval_worker_nums 1 \
    --task_env MT50_task_env \
    --gradient_steps 100 \
    --expert_num 3 \
    --no_cuda \
    --test true \
    --soft true \
    --load_action_path ./fig/VAE_success_1.json \
    --load_from_checkpoint ./fig/VAE_success_1_model.pth

# python starter/EM.py \
#     --n_iter 50 \
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
#     --expert_num 3 \
#     --no_cuda \
#     --test true \
#     --load_action_path ./fig/EM_success_31.json \
#     --load_from_checkpoint ./fig/new_task_success_0_model.pth
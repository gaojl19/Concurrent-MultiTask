python starter/gaussian_baseline.py \
    --n_iter 50 \
    --eval_interval 1 \
    --learning_rate 1e-4 \
    --ep_len 400 \
    --batch_size 64 \
    --train_batch_size 32 \
    --config config/concurrent.json \
    --id MT50_Single_Task \
    --seed 32 \
    --worker_nums 1 \
    --eval_worker_nums 1 \
    --task_env MT50_task_env \
    --gradient_steps 100 \
    --expert_num 3 \
    --no_cuda \
    --test true \
    --load_from_checkpoint ./fig/baseline_task_success_model.pth


# python starter/EM.py \
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
#     --load_from_checkpoint ./fig/EM_success_model.pth
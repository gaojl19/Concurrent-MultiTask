import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    parser.add_argument('--input_dim', type=int, default=0,
                        help='random seed (default: 1)')

    parser.add_argument('--worker_nums', type=int, default=4,
                        help='worker nums')

    parser.add_argument('--eval_worker_nums', type=int, default=2,
                        help='eval worker nums')

    parser.add_argument("--config_file", type=str,   default=None,
                        help="config file", )

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--device", type=int, default=0,
                        help="gpu secification", )

    # tensorboard
    parser.add_argument("--id", type=str,   default=None,
                        help="id for tensorboard", )
    
    # single task learning name
    parser.add_argument("--task_name", type=str, default=None,
                        help="task name for single task training",)
    
    # hyperparameters: for HP tuning
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size for training",)
    
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning rate for BC",)
    
    parser.add_argument("--early_stopping", type=int, default=100,
                        help="early stopping threshold for training BC",)
    
    parser.add_argument("--n_layers", type=int, default=2,
                        help="number of layers for BC networks",)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def get_params(file_name):
    with open(file_name) as f:
        params = json.load(f)
    return params
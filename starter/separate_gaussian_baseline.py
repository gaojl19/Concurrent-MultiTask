import os
import time
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import gym

import sys
# import sys
sys.path.append(".") 

from torch_rl.rl_trainer import RL_Trainer
from agents.bc_agent import MultiHeadAgent
from policy.loaded_gaussian_policy import LoadedGaussianPolicy
from metaworld_utils.concurrent_sawyer import ConcurrentSawyerEnv
from metaworld_utils import SEPARATE_CONCURRENT
from utils.args import get_params
from utils.logger import Logger
from networks.base import MLPBase
from metaworld_utils.meta_env import get_meta_env



class BC_Trainer(object):
    def __init__(self, args, params):
        
        # args and parameters
        self.args = args
        self.params = params
        
        
        # BUILD ENV
        self.device = torch.device("cuda:{}".format(args["device"]) if args["cuda"] else "cpu")
        cls_dicts = {"push": ConcurrentSawyerEnv}
        cls_args = {"push": dict(args=[], kwargs={'obs_type': params['meta_env']['obs_type'], 'task_types': [args["task_types"]]})}
        env_name =  ConcurrentSawyerEnv
        self.env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 

        self.env.seed(args["seed"])
        torch.manual_seed(args["seed"])
        np.random.seed(args["seed"])
        if args["cuda"]:
            torch.backends.cudnn.deterministic=True

    
        self.params['general_setting']['env'] = self.env
        self.params['general_setting']['device'] = self.device
        self.params['net']['base_type']=MLPBase
        
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params = params
        
        # Observation and action shapes
        ob_dim = SEPARATE_CONCURRENT["obs_dim"]
        ac_dim = SEPARATE_CONCURRENT["acs_dim"]
        
        
        # Agent
        agent_params = {
            'n_layers': args['n_layers'],
            'size': args['size'],
            'learning_rate': args['learning_rate'],
            'max_replay_buffer_size': args['max_replay_buffer_size'],
            }
        self.args['agent_class'] = MultiHeadAgent 
        self.args['agent_params'] = agent_params
        self.args['agent_params']['discrete'] = discrete
        self.args['agent_params']['ac_dim'] = ac_dim
        self.args['agent_params']['ob_dim'] = ob_dim
        self.args['agent_params']['head_num'] = 1
        agent = MultiHeadAgent(self.env, self.args['agent_params'], self.params)

        if args["load_from_checkpoint"]:
            agent.actor.policy.load_state_dict(torch.load(args["load_from_checkpoint"], map_location='cpu'))

        
        # EXPERT file path
        # TAG = args["expert_index"]
        # if args["task_type"] == "push_1":
        #     # concurrent learning both push-1 and push-2
        #     expert_file_path = ["../Expert/Concurrent/" + TAG + "push_1.json", "../Expert/Concurrent/" + TAG + "push_3.json"]
        # elif args["task_type"] == "push_2":
        #     expert_file_path = ["../Expert/Concurrent/" + TAG + "push_2.json"]
        # else:
        #     raise NotImplementedError("Invalid task_type!" + args["task_type"])

        
        # LOG PREFIX
        log_prefix = "./fig/separate_gaussian_baseline/" + args["task_types"] + "/"
    
        # RL TRAINER
        self.rl_trainer = RL_Trainer(env = self.env, 
                                     env_cls = cls_dicts, 
                                     env_args = [params["env"], cls_args, params["meta_env"]], 
                                     args = self.args, params = params, 
                                     agent = agent,
                                     input_shape = ob_dim,
                                     expert_num = args["expert_num"],
                                     plot_prefix=log_prefix,
                                     task_types=[args["task_types"]],
                                     mt_flag=False,
                                     idx_flag=True) 
        
    
    def run_training_loop(self, stdscr=None, interface=False):
        if interface:
            self.rl_trainer.run_training_loop(
                n_iter=self.args['n_iter'],
                baseline=False,
                multiple_samples=1,
                stdscr = stdscr
            )
        else:
            self.rl_trainer.run_training_loop(
                n_iter=self.args['n_iter']
            )
    
    def run_test(self):
        self.rl_trainer.agent.actor.eval()
        self.rl_trainer.test_agent()
    
   
def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # customized parameters
    parser.add_argument("--task_types", type=str, default=None,help="task name for single task training: push-1, push-2",)
    parser.add_argument("--task_env", type=str, default=None, help="task to env mapping for single task training: MT10_task_env / MT50_task_env",)
    parser.add_argument("--expert_num", type=int, default=None, help="expert file prefix index", )
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--interface", type=bool, default=False)
    
    # training settings
    parser.add_argument('--ep_len', type=int)
    parser.add_argument('--gradient_steps', type=int, default=1)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)
    parser.add_argument('--render_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=32)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=32)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=400)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)  # LR for supervised learning

  
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    # parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    
    
    parser.add_argument('--worker_nums', type=int, default=4, help='worker nums')
    parser.add_argument('--eval_worker_nums', type=int, default=2,help='eval worker nums')
    parser.add_argument("--config", type=str,   default=None, help="config file", )
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument("--random_init", type=bool, default=False, help="whether use random init when collecting data & evaluating", )
    parser.add_argument("--device", type=int, default=0, help="gpu secification", )
    
    parser.add_argument("--multiple_runs", type=bool, default=False, help="run multiple training loops with different sample sizes", )
    
    # tensorboard
    parser.add_argument("--id", type=str, default=None, help="id for tensorboard", )
    
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.cuda:
        args.device = "cpu"
        
    params = get_params(args.config)
    args = vars(args)
    
    
    # RUN TRAINING
    trainer = BC_Trainer(args, params)
    if args["test"]:
        trainer.run_test()
    else:
        if args["multiple_runs"]:
            trainer.run_multiple_training_loop()
        else:
            trainer.run_training_loop()

if __name__ == "__main__":
    main()
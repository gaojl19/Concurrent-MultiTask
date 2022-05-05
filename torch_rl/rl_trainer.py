from utils.utils import *
from metaworld_utils.meta_env import generate_single_mt_env
from torch_rl.replay_buffer import EnvInfo
from utils.utils import Path
from torch_rl.concurrent_task_collector import ConcurrentCollector

import torch
import numpy as np
import os
import time
from collections import OrderedDict
import seaborn as sns
import pandas as pd
import torch.multiprocessing as mp
from scipy.signal import savgol_filter
import time

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):
    def __init__(self, env, env_cls, env_args, args, params, agent, expert_num, input_shape, task_types, plot_prefix=None, mt_flag=False):
        
        # environment
        self.env = env
        self.env_cls = env_cls
        self.env_args = env_args
        self.env_info = EnvInfo(
            env, params['general_setting']['device'], params['general_setting']['train_render'], params['general_setting']['eval_render'],
            params['general_setting']['epoch_frames'], params['general_setting']['eval_episodes'],
            params['general_setting']['max_episode_frames'], True, None
        )
        self.env_name = params["env_name"]
        
        
        # Arguments
        self.args = args
        self.params = params
        
        # Agent
        self.agent = agent
        
        # other parameters
        self.input_shape = input_shape
        self.expert_num = expert_num
        self.mt_flag = mt_flag
        
        if not os.path.isdir(plot_prefix):
            os.makedirs(plot_prefix)
        
        self.plot_prefix = plot_prefix
        print("plot prefix: ", plot_prefix)
        
        self.expert_env = ConcurrentCollector(
            env=env,
            env_cls=env_cls,
            env_args=env_args,
            env_info=self.env_info,
            device=params['general_setting']['device'],
            max_path_length=self.args["ep_len"],
            min_timesteps_per_batch=self.args['batch_size'],
            input_shape = input_shape,
            task_types=task_types
        )
    
    
    def collect_expert_demo(self):
        # collect trajectories, to be used for training
        
        for i in range(self.expert_num):
            TAG = str(i+1)
            if self.mt_flag == False:
                if self.args["task_types"] == "push-1":   # concurrent learning both push-1 and push-2
                    expert_file_path = ["../Expert/Concurrent/" + TAG + "/push_1.json", "../Expert/Concurrent/" + TAG + "/push_3.json"]
                elif self.args["task_types"] == "push-2":
                    expert_file_path = ["../Expert/Concurrent/" + TAG + "/push_2.json"]
                else:
                    raise NotImplementedError("Invalid task_type!" + self.args["task_types"])
            else:
                expert_file_path = ["../Expert/Concurrent/" + TAG + "/push_1.json", "../Expert/Concurrent/"+ TAG + "/1.json", "../Expert/Concurrent/" + TAG + "/push_2.json", "../Expert/Concurrent/"+ TAG + "/2.json", "../Expert/Concurrent/" + TAG + "/push_3.json"]
            
            training_returns = self.expert_env.sample_expert(action_file=expert_file_path, render=True, log=True, plot_prefix = self.plot_prefix, tag = TAG)
        
            paths, envsteps_this_batch= training_returns
            self.total_envsteps += envsteps_this_batch
            
            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)
                
        print("total training samples: ", self.total_envsteps)


    def run_training_loop(self, n_iter, stdscr=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        loss_curve = []
        agent_success_curve = []
    
    
        # TRAIN
        for itr in range(n_iter):
            print("\n\n-------------------------------- Iteration %i -------------------------------- "%itr)
            start = time.time()
            train_start_time = time.time()
        

            if itr == 0:
                self.collect_expert_demo()
            
            # train agent (using sampled data from replay buffer)
            training_logs = self.train_agent()  # whether or not do alternate training
            
            min_loss = 1000
            for log in training_logs:
                loss_curve.append(log["Training Loss"])
                if min_loss > log["Training Loss"]:
                    min_loss = log["Training Loss"]
        
            train_time = time.time() - train_start_time
            

            # EVALUATION
            eval_start_time = time.time()
            if itr % self.args["eval_interval"] == 0:
                print("\n\n-------------------------------- Evaluating Iteration %i -------------------------------- "%itr)
                render = self.params["general_setting"]["eval_render"]
                
                success_dict = self.expert_env.run_agent(policy=self.agent.actor, render=render, log=True, log_prefix = self.plot_prefix, n_iter=itr)
                agent_success_curve.append(success_dict)
                
                eval_time = time.time() - eval_start_time
                print("training time: ", train_time)
                print("evaluation time: ", eval_time)
                print("epoch time: ", time.time() - start)
                for log in training_logs:
                    print("loss: ", log["Training Loss"])
            
            if min_loss < 0.0001:
                print("\n\n-------------------------------- Training stopped due to early stopping -------------------------------- ")
                print("min loss: ", min_loss)
                break
        
        # TEST
        print("\n\n-------------------------------- Test Results -------------------------------- ")
        render = True
        
        # prepare to save model
        import os.path as osp
        model_file_name="model.pth"
        model_path=osp.join(self.plot_prefix, model_file_name)
        
        success_dict = self.expert_env.run_agent(policy=self.agent.actor, render=render, log=True, log_prefix = self.plot_prefix, n_iter=itr)
        torch.save(self.agent.actor.state_dict(), model_path)
        # else:
        #     success_dict = self.expert_env.run_agent(log_prefix=self.plot_prefix, agent_policy=self.agent.actor.policy, input_shape = self.agent.actor.input_shape, render=render)
        #     torch.save(self.agent.actor.state_dict(), model_path)

        
        # PLOT CURVE
        # plot overall loss curve
        iteration = range(len(loss_curve)-1)
        data = pd.DataFrame(loss_curve[1:], iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("BC--Loss Curve")
        
        fig = ax.get_figure()
        fig.savefig(self.plot_prefix + "Loss_lr_"+str(self.args['learning_rate']*10000)+".png")
        fig.clf()
        
        # plot expert success curve
        # self.plot_success_curve(expert_success_curve, "expert", self.plot_prefix)
        
        # plot agent success curve
        self.plot_success_curve(agent_success_curve, self.plot_prefix)
   
    
    def train_agent(self):
        all_logs = []
 
        for _ in range(self.args['gradient_steps']):

            # sample some data from the data buffer, and train on that batch
            ob_batch, ac_batch = self.agent.sample(self.args['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch)
                
            all_logs.append(train_log)
            
        return all_logs
            
            
    def plot_single_curve(self, curve, tag, plot_prefix, task_name):
        iteration = range(1, len(curve)+1)
        # smooth
        # curve_hat = savgol_filter(curve, 5, 3)
        # data = pd.DataFrame(curve_hat, iteration)
        
        # data = pd.DataFrame(curve, iteration)
        # ax=sns.lineplot(data=data)
        # ax.set_xlabel("Iteration")
        # ax.set_ylabel(tag + " success")
        # ax.set_title(tag + "_" + task_name + " Mean Success Curve")
        # ax.set(ylim=(-0.1, 1.1))
        
        # fig = ax.get_figure()
        # fig.savefig(plot_prefix + task_name + "_"+ tag + "_success_curve.png")
        
        # fig.clf()
        
    def plot_success_curve(self, curve, plot_prefix):
        iteration = range(1, len(curve)+1)
        
        # log success_rate for future usage
        import json
        success_dict = {}
        for key in curve[0].keys():
            success_dict[key] = []
            
        for i in range(len(curve)):
            for key in curve[i].keys():
                success_dict[key].append(curve[i][key])
                
        for key in success_dict.keys():
            success_json = json.dumps(success_dict, sort_keys=False, indent=4)
            f = open(self.plot_prefix + "_success.json", 'w')
            f.write(success_json)
        
            # smooth
            curve_hat = savgol_filter(success_dict[key], 3, 2)
            data = pd.DataFrame(curve_hat, iteration)
            ax=sns.lineplot(data=data)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(str(key) + " success")
            ax.set_title(str(key) + " Mean Success Curve")
            ax.set(ylim=(-0.1, 1.1))
            
            fig = ax.get_figure()
            fig.savefig(plot_prefix + str(key) + "_success_curve.png")
            
            fig.clf()
        
    
    
    def plot_curve(self, curve, tag, plot_prefix):
        iteration = range(1, len(curve)+1)
        
        # smooth
        curve_hat = savgol_filter(curve, 3, 2)
        data = pd.DataFrame(curve_hat, iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(tag)
        ax.set_title(tag)
        ax.set(ylim=(-0.1, 1.1))
        
        fig = ax.get_figure()
        fig.savefig(plot_prefix + tag + "_curve.png")
        
        fig.clf()

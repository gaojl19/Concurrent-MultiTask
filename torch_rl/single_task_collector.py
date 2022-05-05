from curses import KEY_BACKSPACE
from operator import index
from os import lockf
from re import L
from tokenize import Single
import numpy as np
import torch
import time
import copy
import imageio
from utils.utils import *
from metaworld_utils.meta_env import generate_single_mt_env

class SingleCollector():
    def __init__(self, env, env_cls, env_args, env_info, expert_policy, device, max_path_length, min_timesteps_per_batch, embedding_input = [], index_input = None, input_shape = None):
        self.env = copy.deepcopy(env)
        self.env_cls = copy.deepcopy(env_cls)
        self.env_args = copy.deepcopy(env_args)
        self.env_info = copy.deepcopy(env_info) # detach it from other
        self.expert_policy = expert_policy
        self.device = device
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.embedding_input = embedding_input
        self.index_input = index_input
        self.input_shape = expert_policy.ob_dim
        
        self.env_info.env_cls = generate_single_mt_env
        tasks = list(self.env_cls.keys())
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": 1,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }

        for i, task in enumerate(tasks): # currently only 1 task
            env_cls = self.env_cls[task]
            self.task = task
            
            self.env_info.env_rank = i 
            self.env_info.device = "cuda:0"
            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            self.env_info.env_args["env_rank"] = i
            self.env_info.env = self.env_info.env_cls(**self.env_info.env_args)
            
        self.env_info.env.eval()
    
    
    def sample_expert(self, render, render_mode, log, log_prefix, n_iter=0, multiple_samples=1):
        
        print("multiple_samples: ", multiple_samples)
        # sample multiple_successful samples for training
        if multiple_samples > 1:
            paths = self.sample_successful_trajectories(self.expert_policy, multiple_samples, render, render_mode, run_agent=False, log = True, log_prefix = log_prefix)
        
        # only sample once
        else:
            print("sampling task: ", self.task)
            path = self.sample_trajectory(self.expert_policy, render, render_mode, run_agent=False, log = log, log_prefix = log_prefix, n_iter = n_iter)
            paths = [path]
            
        timesteps_this_batch = 0
        for p in paths:
            timesteps_this_batch += len(p["observation"])
        info = None
        # print(self.env_info.env, path["observation"].shape)
        return paths, timesteps_this_batch, info
    
    
    def sample_agent(self, agent_policy, n_sample, render, render_mode, log, log_prefix, n_iter):
        '''
            for Single-task baseline
        '''
        paths = self.sample_n_trajectories(agent_policy, n_sample, render, render_mode, run_agent=True, log = log, log_prefix = log_prefix, n_iter = n_iter)
        success = 0
        for path in paths:
            if path["success"] == True:
                success += 1
        
        mean_success_rate = success/n_sample
        return mean_success_rate
    
    def sample_embedding_agent(self, agent_policy, n_sample, render, render_mode, log, log_prefix, n_iter):
        '''
            for MT50 baseline
        '''
        paths = self.sample_n_trajectories(agent_policy, n_sample, render, render_mode, run_agent=True, log = log, log_prefix = log_prefix, n_iter = n_iter, use_embedding=True)
        success = 0
        for path in paths:
            if path["success"] == True:
                success += 1
        
        mean_success_rate = success/n_sample
        return mean_success_rate
    
    
    # the policy gradient should be frozen before sending into this function
    def sample_trajectory(self, policy, render=False, render_mode=('rgb_array'), run_agent=False, log = False, log_prefix = "./", n_iter=0, use_embedding=False):
        
        # initialize env for the beginning of a new rollout
        env = self.env_info.env
        env.eval()
        ob = env.reset()
        log_info = "initial ob: " + str(ob) + "\n"
        
        # init vars
        obs, acs, rewards, next_obs, terminals, image_obs, embedding_input, index_input = [], [], [], [], [], [], [], []
        steps = 0
        done = False
        success = 0
        
        while True:
            embedding_input.append(self.embedding_input)
            index_input.append(self.index_input)
            
            # query the policy's get_action function
            if not run_agent:
                # use the most recent ob to decide what to do
                ob = ob[:self.input_shape]
                obs.append(ob)
                act = policy.get_action(torch.Tensor(ob).to(self.device).unsqueeze(0), self.device)
                # log_info += "expert:" + str(act) + "\n"
            
            else:
                # use the most recent ob to decide what to do
                if use_embedding:
                    ob = ob[:policy.input_shape-self.embedding_input.shape[1]]
                    obs.append(ob)
                    ob = torch.Tensor(np.concatenate((ob, self.embedding_input.squeeze())))
                else:
                    ob = ob[:policy.input_shape]
                    obs.append(ob)
                act = policy.get_action(torch.Tensor(ob).to(self.device).unsqueeze(0)).detach().cpu().numpy()
                act = np.squeeze(act)
                # log_info += "agent:" + str(act) + "\n"
                
            acs.append(act)
            
            # take that action and record results
            ob, r, done, info = env.step(act)
            if not run_agent:
                ob = ob[:self.input_shape]
            else:
                if use_embedding:
                    ob = ob[:policy.input_shape-self.embedding_input.shape[1]]
                else:
                    ob = ob[:policy.input_shape]
            
            # record result of taking that action
            steps += 1
            next_obs.append(ob)
            rewards.append(r)
            
            # only support rbg_array mode currently
            if render:
                # image_obs.append(env.render(mode='rgb_array'))
                image = env.get_image(400,400,'leftview')
                image_obs.append(image)
            
            success = max(success, info["success"])

            # end the rollout if the rollout ended
            rollout_done = True if (done or steps>=self.max_path_length) else False
            terminals.append(rollout_done)

            if rollout_done:
                break
        
        if not run_agent:
            log_info += "expert_success: " + str(success) + "\n"
            log_info += "path_length: " + str(len(acs)) + "\n"
        else:
            log_info += "agent_success: " + str(success) + "\n"
            
        if len(image_obs)>0:
            if run_agent == True:
                imageio.mimsave(log_prefix + str(n_iter) + "_agent.gif", image_obs)
            else:
                imageio.mimsave(log_prefix + str(n_iter) + "_expert.gif", image_obs)
            
        if log == True:
            print(log_info)
            
        return Path(obs, image_obs, acs, rewards, next_obs, terminals, success, embedding_input, index_input)


    def sample_trajectories(self, policy, render=False, render_mode=('rgb_array'), run_agent=False, log=False, log_prefix = "./"):
        """
            Collect rollouts until we have collected min_timesteps_per_batch steps.
            
            Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
            Hint2: use get_pathlength to count the timesteps collected in each path
        """
        timesteps_this_batch = 0
        paths = []
        while timesteps_this_batch < self.min_timesteps_per_batch:
            new_path = self.sample_trajectory(policy=policy, render=render, render_mode=render_mode, run_agent=run_agent, log=log, log_prefix=log_prefix)
            paths.append(new_path)
            timesteps_this_batch += get_pathlength(new_path)

        return paths, timesteps_this_batch

    def sample_n_trajectories(self, policy, ntraj, render=False, render_mode=('rgb_array'), run_agent=False, log = False, log_prefix = "./", n_iter = 0, use_embedding=False):
        """
            Collect ntraj rollouts.
            
            Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        """
        paths = []

        for i in range(ntraj):
            if i == int(ntraj/2):
                paths.append(self.sample_trajectory(policy=policy, render=render, render_mode=render_mode, run_agent=run_agent, log=log, log_prefix=log_prefix, n_iter=n_iter, use_embedding=use_embedding))
            else:
                paths.append(self.sample_trajectory(policy=policy, render=False, render_mode=render_mode, run_agent=run_agent, log=log, log_prefix=log_prefix, n_iter=n_iter))
        return paths
    
    
    def sample_successful_trajectories(self, policy, ntraj, render=False, render_mode=('rgb_array'), run_agent=False, log = False, log_prefix = "./", n_iter = 0):
        """
            Collect ntraj successful rollouts
            
        """
        paths = []
        cnt = 0
        
        for i in range(1000):
            new_path = self.sample_trajectory(policy=policy, render=False, render_mode=render_mode, run_agent=run_agent, log=log, log_prefix=log_prefix, n_iter=n_iter)
            if new_path["success"] == 1.0:
                paths.append(new_path)
                cnt += 1
                if cnt == ntraj:
                    print("successful trajectory collected: ", cnt)
                    return paths
        print("successful trajectory collected: ", cnt)
        return paths

    
    


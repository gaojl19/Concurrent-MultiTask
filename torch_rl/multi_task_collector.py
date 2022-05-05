from audioop import mul
from lib2to3.pgen2.token import N_TOKENS
from operator import index

from jinja2 import TemplateNotFound
from utils.utils import *
from metaworld_utils.meta_env import generate_single_mt_env
from torch_rl.replay_buffer import EnvInfo
from policy.continuous_policy import MultiHeadGuassianContPolicy, EmbeddingGuassianContPolicyBase
from utils.utils import Path
from agents.bc_agent import SoftModuleAgent

from sklearn.manifold import TSNE
import seaborn as sns

import torch
import numpy as np
import os
import time
from collections import OrderedDict
import seaborn as sns
import pandas as pd
import torch.multiprocessing as mp
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT, EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS
from metaworld_utils.meta_env import get_meta_env
from metaworld_utils.customize_env_dict import DIVERSE_MT10_CLS_DICT, DIVERSE_MT10_ARGS_KWARGS
from metaworld_utils.customize_env_dict import SIMILAR_MT10_CLS_DICT, SIMILAR_MT10_ARGS_KWARGS
from metaworld_utils.customize_env_dict import FAIL_MT10_CLS_DICT, FAIL_MT10_ARGS_KWARGS
from metaworld_utils.customize_env_dict import MEDIUM_MT10_CLS_DICT, MEDIUM_MT10_ARGS_KWARGS
from metaworld_utils.customize_env_dict import HARD_MT10_CLS_DICT, HARD_MT10_ARGS_KWARGS
from metaworld_utils.customize_env_dict import MT40_CLS_DICT, MT40_ARGS_KWARGS
from torch_rl.single_task_collector import SingleCollector



class MT10SingleCollector():
    '''
        Create 10 single task environment, sample paths from single-task expert policy
    '''
    def __init__(self, env_cls, env_args, env_info, expert_dict, device, max_path_length, min_timesteps_per_batch, params, input_shape):
        self.tasks = list(env_cls.keys())
        self.device = device
        self.task_collector = {}
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch/10
        self.input_shape = input_shape
        
        for i, task in enumerate(self.tasks):
            cls_dicts = {task: EASY_MODE_CLS_DICT[task]}
            cls_args = {task: EASY_MODE_ARGS_KWARGS[task]}
            env_name =  EASY_MODE_CLS_DICT[task]
            
            # overwrite random init part
            cls_args[task]['kwargs']['obs_type'] = params['meta_env']['obs_type']
            cls_args[task]['kwargs']['random_init'] = params['meta_env']['random_init']
    
            # env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
            env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
            
            # create embedding
            embedding_input = torch.zeros(10)
            embedding_input[i] = 1
            embedding_input = embedding_input.unsqueeze(0).to(self.device)
            
            # create index
            index_input = torch.Tensor([[i]]).to(env_info.device).long()
        
            if task in expert_dict.keys():
                self.task_collector[task] = SingleCollector(
                    env=env, 
                    env_cls=cls_dicts, 
                    env_args=env_args,
                    env_info=env_info,
                    expert_policy=expert_dict[task],
                    device=device,
                    max_path_length=self.max_path_length,
                    min_timesteps_per_batch=self.min_timesteps_per_batch,
                    embedding_input=embedding_input,
                    index_input = index_input,
                    input_shape=input_shape)
        
    
    def sample_expert(self, render, render_mode, log, log_prefix, multiple_samples):
        '''
            serialized sample from 10 environment
        '''
        paths = []
        timesteps_this_batch = 0
        info = {}
        success = 0
        for task in self.task_collector.keys():
            collector = self.task_collector[task]
            new_path, timesteps, infos = collector.sample_expert(render, render_mode, log, log_prefix, multiple_samples=multiple_samples)
            # modify observations
            if self.input_shape > len(new_path[0]["observation"][0]):
                new_path[0]["observation"] = [np.append(ob, ob[6:]) if len(ob)==9 else ob for ob in new_path[0]["observation"]]
                new_path[0]["next_observation"] = [np.append(ob, ob[6:]) if len(ob)==9 else ob for ob in new_path[0]["next_observation"]]
            
            paths += new_path
            timesteps_this_batch += timesteps
            info[task + "_success_rate"] = new_path[0]["success"]
            success += new_path[0]["success"]
        
        info["mean_success_rate"] = success / len(self.task_collector )
        print(info)
        return paths, timesteps_this_batch, info
    
    
    def sample_agent(self, agent_policy, n_sample, render, render_mode, log, log_prefix, n_iter=0):
        '''
            serialized sample from 50 environment
        '''
        info = {}
        success = 0
        for task in self.task_collector.keys():
            # prefix = log_prefix + "/" + task + "/"
            collector = self.task_collector[task]
            success_rate = collector.sample_embedding_agent(agent_policy, n_sample, render, render_mode, log, log_prefix, n_iter)
            
            info[task + "_success_rate"] = success_rate
            success += success_rate
        
        info["mean_success_rate"] = success / len(self.task_collector )
        print(info)
        return info
    

class MT10DiverseCollector(MT10SingleCollector):
    '''
        Diverse version, 10 tasks
    '''
    def __init__(self, env_cls, env_args, env_info, expert_dict, device, max_path_length, min_timesteps_per_batch, params, input_shape):
        self.tasks = list(env_cls.keys())
        self.device = device
        self.task_collector = {}
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch/10
        self.input_shape = input_shape
        
        for i, task in enumerate(self.tasks):
            cls_dicts = {task: DIVERSE_MT10_CLS_DICT[task]}
            cls_args = {task: DIVERSE_MT10_ARGS_KWARGS[task]}
            env_name =  DIVERSE_MT10_CLS_DICT[task]
            
            # overwrite random init part
            cls_args[task]['kwargs']['obs_type'] = params['meta_env']['obs_type']
            cls_args[task]['kwargs']['random_init'] = params['meta_env']['random_init']
    
            # env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
            env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
            
            # create embedding
            embedding_input = torch.zeros(10)
            embedding_input[i] = 1
            embedding_input = embedding_input.unsqueeze(0).to(self.device)
            
            # create index
            index_input = torch.Tensor([[i]]).to(env_info.device).long()
        
            if task in expert_dict.keys():
                self.task_collector[task] = SingleCollector(
                    env=env, 
                    env_cls=cls_dicts, 
                    env_args=env_args,
                    env_info=env_info,
                    expert_policy=expert_dict[task],
                    device=device,
                    max_path_length=self.max_path_length,
                    min_timesteps_per_batch=self.min_timesteps_per_batch,
                    embedding_input=embedding_input,
                    index_input=index_input,
                    input_shape=input_shape)

class MT10SimilarCollector(MT10SingleCollector):
    '''
        Similar version, 10 tasks
    '''
    def __init__(self, env_cls, env_args, env_info, expert_dict, device, max_path_length, min_timesteps_per_batch, params, input_shape):
        self.tasks = list(env_cls.keys())
        self.device = device
        self.task_collector = {}
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch/10
        self.input_shape = input_shape
        
        for i, task in enumerate(self.tasks):
            cls_dicts = {task: SIMILAR_MT10_CLS_DICT[task]}
            cls_args = {task: SIMILAR_MT10_ARGS_KWARGS[task]}
            env_name =  SIMILAR_MT10_CLS_DICT[task]
            
            # overwrite random init part
            cls_args[task]['kwargs']['obs_type'] = params['meta_env']['obs_type']
            cls_args[task]['kwargs']['random_init'] = params['meta_env']['random_init']
    
            # env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
            env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
            
            # create embedding
            embedding_input = torch.zeros(10)
            embedding_input[i] = 1
            embedding_input = embedding_input.unsqueeze(0).to(self.device)
            
            # create index
            index_input = torch.Tensor([[i]]).to(env_info.device).long()
        
            if task in expert_dict.keys():
                self.task_collector[task] = SingleCollector(
                    env=env, 
                    env_cls=cls_dicts, 
                    env_args=env_args,
                    env_info=env_info,
                    expert_policy=expert_dict[task],
                    device=device,
                    max_path_length=self.max_path_length,
                    min_timesteps_per_batch=self.min_timesteps_per_batch,
                    embedding_input=embedding_input,
                    index_input=index_input,
                    input_shape=input_shape)
                
class MT10FailCollector(MT10SingleCollector):
    '''
        Partially failed version, 10 tasks
    '''
    def __init__(self, env_cls, env_args, env_info, expert_dict, device, max_path_length, min_timesteps_per_batch, params, input_shape):
        self.tasks = list(env_cls.keys())
        self.device = device
        self.task_collector = {}
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch/10
        self.input_shape = input_shape
        
        for i, task in enumerate(self.tasks):
            cls_dicts = {task: FAIL_MT10_CLS_DICT[task]}
            cls_args = {task: FAIL_MT10_ARGS_KWARGS[task]}
            env_name = FAIL_MT10_CLS_DICT[task]
            
            # overwrite random init part
            cls_args[task]['kwargs']['obs_type'] = params['meta_env']['obs_type']
            cls_args[task]['kwargs']['random_init'] = params['meta_env']['random_init']
    
            # env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
            env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
            
            # create embedding
            embedding_input = torch.zeros(10)
            embedding_input[i] = 1
            embedding_input = embedding_input.unsqueeze(0).to(self.device)
            
            # create index
            index_input = torch.Tensor([[i]]).to(env_info.device).long()
        
            if task in expert_dict.keys():
                self.task_collector[task] = SingleCollector(
                    env=env, 
                    env_cls=cls_dicts, 
                    env_args=env_args,
                    env_info=env_info,
                    expert_policy=expert_dict[task],
                    device=device,
                    max_path_length=self.max_path_length,
                    min_timesteps_per_batch=self.min_timesteps_per_batch,
                    embedding_input=embedding_input,
                    index_input=index_input,
                    input_shape=input_shape) 

class MT10MediumCollector(MT10SingleCollector):
    '''
        Medium version, 10 tasks with 7 successful and 3 unsuccessful
    '''
    def __init__(self, env_cls, env_args, env_info, expert_dict, device, max_path_length, min_timesteps_per_batch, params, input_shape):
        self.tasks = list(env_cls.keys())
        self.device = device
        self.task_collector = {}
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch/10
        self.input_shape = input_shape
        
        for i, task in enumerate(self.tasks):
            cls_dicts = {task: MEDIUM_MT10_CLS_DICT[task]}
            cls_args = {task: MEDIUM_MT10_ARGS_KWARGS[task]}
            env_name =  MEDIUM_MT10_CLS_DICT[task]
            
            # overwrite random init part
            cls_args[task]['kwargs']['obs_type'] = params['meta_env']['obs_type']
            cls_args[task]['kwargs']['random_init'] = params['meta_env']['random_init']
    
            # env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
            env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
            
            # create embedding
            embedding_input = torch.zeros(10)
            embedding_input[i] = 1
            embedding_input = embedding_input.unsqueeze(0).to(self.device)
            
            # create index
            index_input = torch.Tensor([[i]]).to(env_info.device).long()
        
            if task in expert_dict.keys():
                self.task_collector[task] = SingleCollector(
                    env=env, 
                    env_cls=cls_dicts, 
                    env_args=env_args,
                    env_info=env_info,
                    expert_policy=expert_dict[task],
                    device=device,
                    max_path_length=self.max_path_length,
                    min_timesteps_per_batch=self.min_timesteps_per_batch,
                    embedding_input=embedding_input,
                    index_input=index_input,
                    input_shape=input_shape)
                
                
class MT10HardCollector(MT10SingleCollector):
    '''
        Hard version, 10 tasks with 5 successful and 5 unsuccessful
    '''
    def __init__(self, env_cls, env_args, env_info, expert_dict, device, max_path_length, min_timesteps_per_batch, params, input_shape):
        self.tasks = list(env_cls.keys())
        self.device = device
        self.task_collector = {}
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch/10
        self.input_shape = input_shape
        
        for i, task in enumerate(self.tasks):
            cls_dicts = {task: HARD_MT10_CLS_DICT[task]}
            cls_args = {task: HARD_MT10_ARGS_KWARGS[task]}
            env_name =  HARD_MT10_CLS_DICT[task]
            
            # overwrite random init part
            cls_args[task]['kwargs']['obs_type'] = params['meta_env']['obs_type']
            cls_args[task]['kwargs']['random_init'] = params['meta_env']['random_init']
    
            # env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
            env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
            
            # create embedding
            embedding_input = torch.zeros(10)
            embedding_input[i] = 1
            embedding_input = embedding_input.unsqueeze(0).to(self.device)
            
            # create index
            index_input = torch.Tensor([[i]]).to(env_info.device).long()
        
            if task in expert_dict.keys():
                self.task_collector[task] = SingleCollector(
                    env=env, 
                    env_cls=cls_dicts, 
                    env_args=env_args,
                    env_info=env_info,
                    expert_policy=expert_dict[task],
                    device=device,
                    max_path_length=self.max_path_length,
                    min_timesteps_per_batch=self.min_timesteps_per_batch,
                    embedding_input=embedding_input,
                    index_input=index_input,
                    input_shape=input_shape)
                
                
                
class MT40Collector(MT10SingleCollector):
    '''
        40 tasks with all successful expert data
    '''
    def __init__(self, env_cls, env_args, env_info, expert_dict, device, max_path_length, min_timesteps_per_batch, params, input_shape):
        self.tasks = list(env_cls.keys())
        self.device = device
        self.task_collector = {}
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch/40
        self.input_shape = input_shape
        
        for i, task in enumerate(self.tasks):
            cls_dicts = {task: MT40_CLS_DICT[task]}
            cls_args = {task: MT40_ARGS_KWARGS[task]}
            env_name =  MT40_CLS_DICT[task]
            
            # overwrite random init part
            cls_args[task]['kwargs']['obs_type'] = params['meta_env']['obs_type']
            cls_args[task]['kwargs']['random_init'] = params['meta_env']['random_init']
    
            # env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
            env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
            
            # create embedding
            embedding_input = torch.zeros(40)
            embedding_input[i] = 1
            embedding_input = embedding_input.unsqueeze(0).to(self.device)
            
            # create index
            index_input = torch.Tensor([[i]]).to(env_info.device).long()
        
            if task in expert_dict.keys():
                print(task)
                self.task_collector[task] = SingleCollector(
                    env=env, 
                    env_cls=cls_dicts, 
                    env_args=env_args,
                    env_info=env_info,
                    expert_policy=expert_dict[task],
                    device=device,
                    max_path_length=self.max_path_length,
                    min_timesteps_per_batch=self.min_timesteps_per_batch,
                    embedding_input=embedding_input,
                    index_input=index_input,
                    input_shape=input_shape)
    
    
class MT50SingleCollector():
    '''
        Create 50 single task environment, sample paths from single-task expert policy
    '''
    def __init__(self, env_cls, env_args, env_info, expert_dict, device, max_path_length, min_timesteps_per_batch, params, input_shape):
        self.tasks = list(env_cls.keys())
        self.device = device
        self.task_collector = {}
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch/50
        self.input_shape = input_shape
        
        for i, task in enumerate(self.tasks):
            if task in HARD_MODE_CLS_DICT['train'].keys():     # 45 tasks
                cls_dicts = {task: HARD_MODE_CLS_DICT['train'][task]}
                cls_args = {task: HARD_MODE_ARGS_KWARGS['train'][task]}
                env_name =  HARD_MODE_CLS_DICT['train'][task]
            else:
                cls_dicts = {task: HARD_MODE_CLS_DICT['test'][task]}
                cls_args = {task: HARD_MODE_ARGS_KWARGS['test'][task]}
                env_name =  HARD_MODE_CLS_DICT['test'][task]
            
            
            # overwrite random init part
            cls_args[task]['kwargs']['obs_type'] = params['meta_env']['obs_type']
            cls_args[task]['kwargs']['random_init'] = params['meta_env']['random_init']
    
            # env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
            env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
            
            # create embedding
            embedding_input = torch.zeros(50)
            embedding_input[i] = 1
            embedding_input = embedding_input.unsqueeze(0).to(self.device)
            
            # create index
            index_input = torch.Tensor([[i]]).to(env_info.device).long()
            
            if task in expert_dict.keys():
                self.task_collector[task] = SingleCollector(
                    env=env, 
                    env_cls=cls_dicts, 
                    env_args=env_args,
                    env_info=env_info,
                    expert_policy=expert_dict[task],
                    device=device,
                    max_path_length=self.max_path_length,
                    min_timesteps_per_batch=self.min_timesteps_per_batch,
                    embedding_input=embedding_input,
                    index_input=index_input,
                    input_shape=self.input_shape)
            
                
    
    def sample_expert(self, render, render_mode, log, log_prefix, multiple_samples):
        '''
            serialized sample from 50 environment
        '''
        paths = []
        timesteps_this_batch = 0
        info = {}
        success = 0
        for task in self.task_collector.keys():
            # prefix = log_prefix + "/" + task + "/"
            collector = self.task_collector[task]
            new_path, timesteps, infos = collector.sample_expert(render, render_mode, log, log_prefix, multiple_samples=multiple_samples)
            
            # modify observations
            new_path[0]["observation"] = [np.append(ob, ob[6:]) if len(ob)==9 else ob for ob in new_path[0]["observation"]]
            new_path[0]["next_observation"] = [np.append(ob, ob[6:]) if len(ob)==9 else ob for ob in new_path[0]["next_observation"]]
            
            # for ob in new_path["observation"]:
            #     print(ob)
            
            paths += new_path
            timesteps_this_batch += timesteps
            info[task + "_success_rate"] = new_path[0]["success"]
            success += new_path[0]["success"]
        
        info["mean_success_rate"] = success / len(self.task_collector )
        print(info)
        return paths, timesteps_this_batch, info
    
    
    def sample_agent(self, agent_policy, n_sample, render, render_mode, log, log_prefix):
        '''
            serialized sample from 50 environment, baseline
        '''
        info = {}
        success = 0
        for task in self.task_collector.keys():
            # prefix = log_prefix + "/" + task + "/"
            collector = self.task_collector[task]
            success_rate = collector.sample_embedding_agent(agent_policy, n_sample, render, render_mode, log, log_prefix)
            
            info[task + "_success_rate"] = success_rate
            success += success_rate
        
        info["mean_success_rate"] = success / len(self.task_collector )
        print(info)
        return info
    
    
class MTEnvCollector():
    '''
        create MT10/50 environment, sample paths from multi-task policy(usually agent policy for evaluation) 
    '''
    def __init__(self, env, env_cls, env_args, env_info, args, params, example_embedding, plot_prefix):
        self.env = env
        self.env_cls = env_cls
        self.env_args = env_args
        self.env_info = copy.deepcopy(env_info)
        self.args = args
        self.params = params
        self.example_embedding = example_embedding
        self.epochs = args["n_iter"]
        self.plot_prefix=plot_prefix
        
        # multi-processing
        self.manager = mp.Manager()
    
    def sample_agent(self, log_prefix, agent_policy, input_shape, render, plot_weights=False):
        
        self.build_Multi_task_env(agent_policy=agent_policy, input_shape=input_shape, render=render, return_weights = plot_weights)
        active_task_counts = 0
        tasks_result = []
        mean_success_rate = 0
        images = {}
        weights = {}
        
        for res in self.results:
            active_task_counts += 1
            mean_success_rate += res["mean_success_rate"]
            tasks_result.append((res["task_name"], res["mean_success_rate"]))
            if len(res["image_obs"]) >0 :
                images[res["task_name"]] = res["image_obs"]
            weights[res["task_name"]] = res['weights']
        
        if len(images) > 0:
            for task_name in images.keys():
                imageio.mimsave(log_prefix + task_name + "_agent.gif", images[task_name])
                
        if plot_weights:
            # self.plot_TSNE(weights=weights)
            self.visualize_single_weights(weights=weights, agent_policy=agent_policy)
    
        tasks_result.sort()
        dic = OrderedDict()
        for task_name, success_rate in tasks_result:
            dic[task_name + "_success_rate"] = success_rate
        
        
        # for p in self.eval_workers:
        #     p.join()
            
        # save expert success_rate and reward infos
        del images
        dic['mean_success_rate'] = mean_success_rate / active_task_counts       
        return dic
    
    def collect_next_state(self, action_dict, input_shape):
        
        self.eval_workers = []
        self.eval_worker_nums = self.env.num_tasks
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)
        
        tasks = list(self.env_cls.keys())
        self.shared_dict = self.manager.dict()
                
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        # print(self.env.num_tasks)
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2],
            "env_name": self.params["env_name"],
        }
        
        self.next_obs = {}
        for i, task in enumerate(tasks):
            self.next_obs[i+1] = []

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i
            
            # create task_embeddings
            embedding_input = torch.zeros(self.env_info.num_tasks)
            embedding_input[self.env_info.env_rank] = 1
            embedding_input = embedding_input.unsqueeze(0).to(self.env_info.device)
            
            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])
            self.env_info.env_args["env_rank"] = i

            # Rebuild Env
            self.env_info.env = self.env_info.env_cls(**self.env_info.env_args)
            norm_obs_flag = self.env_info.env_args["env_params"]["obs_norm"]
            # print("normalize observation: ", norm_obs_flag)
            
            actions = action_dict[i+1]
            
            for act in actions:
                next_ob, r, done, info = self.env_info.env.step(act.detach().numpy())
                # mask out the last 3 dimensions
                if len(next_ob) != input_shape:
                    next_ob = next_ob[:input_shape]
                self.next_obs[i+1].append(next_ob)
        
        return self.next_obs


    def build_Multi_task_env(self, agent_policy, input_shape, render=False, return_weights=False):
    
        self.eval_workers = []
        self.eval_worker_nums = self.env.num_tasks
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)
        
        tasks = list(self.env_cls.keys())
        self.shared_dict = self.manager.dict()
                
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        print(self.env.num_tasks)
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2],
            "env_name": self.params["env_name"],
        }
        
        self.results = []

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            start_epoch = 0
            if "start_epoch" in self.env_info.env_args["task_args"]:
                # start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            # else:
                # start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            result = self.evaluate(agent_policy=agent_policy,
                          env_info=self.env_info,
                          eval_episode=self.params["general_setting"]["eval_episodes"],
                          max_frame=self.params["general_setting"]["max_episode_frames"],
                          task_name=task,
                          shared_dict=self.shared_dict,
                          render=render,
                          input_shape=input_shape,
                          return_weights=return_weights)
            
            self.results.append(result)
            
    
    def evaluate(self, agent_policy, env_info, eval_episode, max_frame, task_name, shared_dict, render, input_shape, return_weights=False):
        
        # pf = copy.deepcopy(agent_policy).to(env_info.device)
        # pf.eval()
        agent_policy.eval()
        pf = agent_policy
        
        idx_flag = isinstance(pf, MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, EmbeddingGuassianContPolicyBase)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)
        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        env_info.env.eval()
        env_info.env._reward_scale = 1
        round = 0
        success = 0
        rew = 0
        image_obs = []
        
        # print("max episode frames: ", env_info.max_episode_frames)
        for i in range(eval_episode):
            if norm_obs_flag:
                env_info.env._obs_mean = shared_dict[task_name]["obs_mean"]
                env_info.env._obs_var = shared_dict[task_name]["obs_var"]
            
            # initialize
            acs = []
            done = False
            weights = []
        
            eval_ob = env_info.env.reset()
            task_idx = env_info.env_rank
            current_success = 0
            current_step = 0
            
            while not done:  
                # mask out the last 3 dimensions
                if len(eval_ob) != input_shape:
                    eval_ob = eval_ob[:input_shape]
                
                if idx_flag:
                    idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                    if embedding_flag:
                        # create embedding
                        embedding_input = torch.zeros(env_info.num_tasks)
                        embedding_input[env_info.env_rank] = 1
                        embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                        
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0),
                            embedding_input, [task_idx] )
                    else:
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), idx_input )
                    
                else:
                    if embedding_flag:
                        # create embedding
                        embedding_input = torch.zeros(env_info.num_tasks)
                        embedding_input[env_info.env_rank] = 1
                        embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                        
                        if return_weights:
                            act, general_weights = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), embedding_input, return_weights=return_weights)
                            weights.append(general_weights)
                        else:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), embedding_input)

                    else:
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))
                
                acs.append(act)

                eval_ob, r, done, info = env_info.env.step( act )
                rew += r
                current_success = max(current_success, info["success"])
                current_step += 1
                
                # render
                if render==True:
                    if i == int(eval_episode/2):
                        image = env_info.env.get_image(400,400,"leftview")
                        # print(env_info.env.get_image(400,400,"frontview"))
                        image_obs.append(image)
                    
                done = False
                if current_step > max_frame:
                    break
            
            success += current_success
            round += 1

        success_rate = success / round
        
        return { 
            'image_obs': image_obs, 
            'mean_success_rate': success_rate, 
            'task_name': task_name,
            'weights': weights
        }
        
    @staticmethod
    def eval_worker_process(shared_pf, env_info, shared_que, start_barrier, eval_episode, start_epoch, max_frame, task_name, shared_dict):
        '''
            1. we only need to return one final dict, with 
               1) task_name
               2) one render rgb image
               3) average_success_rate over eval_episode
        '''
        pf = copy.deepcopy(shared_pf).to(env_info.device)
        pf.eval()
        
        
        idx_flag = isinstance(pf, MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, EmbeddingGuassianContPolicyBase)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)
        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        env_info.env.eval()
        env_info.env._reward_scale = 1
        round = 0
        success = 0
        rew = 0
        
        for i in range(eval_episode):
            if norm_obs_flag:
                env_info.env._obs_mean = shared_dict[task_name]["obs_mean"]
                env_info.env._obs_var = shared_dict[task_name]["obs_var"]
                    
            
            # initialize
            acs = []
            image_obs = []
            done = False
        
            eval_ob = env_info.env.reset()
            task_idx = env_info.env_rank
            current_success = 0
            current_step = 0
            while not done:
                if idx_flag:
                    idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                    if embedding_flag:
                        # create embedding
                        embedding_input = torch.zeros(env_info.num_tasks)
                        embedding_input[env_info.env_rank] = 1
                        embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                        
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0),
                            embedding_input, [task_idx] )
                    else:
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), idx_input )
                    
                else:
                    if embedding_flag:
                        # create embedding
                        embedding_input = torch.zeros(env_info.num_tasks)
                        embedding_input[env_info.env_rank] = 1
                        embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                        
                        # mask out the last 3 dimensions
                        eval_ob = eval_ob[:9]
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), embedding_input)
                    else:
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))
                
                acs.append(act)

                eval_ob, r, done, info = env_info.env.step( act )
                rew += r
                current_success = max(current_success, info["success"])
                current_step += 1
                
                # render
                if env_info.eval_render:
                    if i == int(eval_episode/2):
                        image = env_info.env.get_image(400,400,"leftview")
                        image_obs.append(image)
                    
                done = False
                if current_step > max_frame:
                    break
            
            success += current_success
            round += 1

        success_rate = success / round
        shared_que.put({ 
            'image_obs': image_obs, 
            'mean_success_rate': success_rate, 
            'task_name': task_name
        })


    def plot_TSNE(self, weights):
        '''
            plot t-sne of modular weights
        '''
        palette = sns.color_palette("bright", 10)
        X = []
        y = []
        for task_name in weights.keys():
            for w in weights[task_name]:
                X.append(torch.reshape(w[0], (-1,)))
                y.append(task_name)
        X = torch.stack(X)
        print(X.shape)
        
        # using scikit-learn package
        tsne = TSNE()
        X_embedded = tsne.fit_transform(X)  
        sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
        fig = sns_plot.get_figure()
        fig.savefig(self.plot_prefix + "_tsne.png")
        
    
    
    def visualize_weights(self, weights):
        '''
            visualize weights between modules
        '''
        
        weight_dict = {}
        for task_name in weights.keys():
            X = torch.zeros(1, weights[task_name][0][0].shape[1] * weights[task_name][0][0].shape[2])
            Y = []
            cnt = 0
            for w in weights[task_name]:
                new_value = torch.reshape(w[0], (-1,))
                new_value = torch.nn.functional.normalize(new_value, p=1, dim=0)
                X = torch.add(X, new_value)
                cnt += 1
                # just choose one sample
                break 
            weight_dict[task_name] = X.numpy().tolist()
        
        import json
        weight_json = json.dumps(weight_dict,sort_keys=False, indent=4)
        f = open(self.plot_prefix + "_weight.json", 'w')
        f.write(weight_json)
        
    
    def visualize_single_weights(self, weights, agent_policy):
        '''
            visualize weights between modules
        '''
        import json
        for task_name in weights.keys():
            cnt = 0
            weight_dict = {}
            for w in weights[task_name]:
                w = torch.stack(w, dim=0).view(-1)
                # new_value = torch.reshape(w[0], (-1,))

                weight_dict[cnt] = w.tolist()
                cnt += 1
            # print(task_name)
                
            weight_json = json.dumps(weight_dict,sort_keys=False, indent=4)
            f = open(self.plot_prefix + task_name + "_weight.json", 'w')
            f.write(weight_json)
    

        # # save model
        # import os.path as osp
        # model_file_name="model.pth"
        # model_path=osp.join(self.plot_prefix, model_file_name)
        # torch.save(agent_policy.state_dict(), model_path)
    
   
    
        
        

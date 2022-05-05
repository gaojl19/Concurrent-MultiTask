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


class ConcurrentCollector():
    def __init__(self, env, env_cls, env_args, env_info, device, max_path_length, min_timesteps_per_batch, input_shape, embedding_input = [], index_input = None):
        self.env = copy.deepcopy(env)
        self.env_cls = copy.deepcopy(env_cls)
        self.env_args = copy.deepcopy(env_args)
        self.env_info = copy.deepcopy(env_info) # detach it from other
        self.device = device
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.embedding_input = embedding_input
        self.index_input = index_input
        self.input_shape = input_shape
        
        
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

    
    
    # the policy gradient should be frozen before sending into this function
    def sample_expert(self, action_file, render=True, log=True, tag=None):  
        
        render=True
        log_info = ""
        
        # initialize env for the beginning of a new rollout
        env = self.env_info.env
        env.eval()
        ob = env.reset()
        log_info += "initial obs: "+ str(ob) + "\n"
        # TODO: separate obs for different tasks
        # ob = np.concatenate((ob[:6], ob[9:12]))
        
        # init vars
        obs, acs, next_obs, terminals, image_obs_front, image_obs_left, embedding_input, index_input = [], [], [], [], [], [], [], []
        steps = 0
        done = False
        
        success_push_1 = 0
        success_push_2 = 0
        push_dist1 = 1
        push_dist2 = 1
        
        # read all the actions
        import json
        action_list = []
        action_sequence = []
        
        # for test
        x, y, z = [], [], []
        
        # do separate steps
        for file in action_file:
            initial_ob = ob
            print(initial_ob)

            with open(file, 'r') as fin:
                ac = json.load(fin)["actions"]
                action_list.append(ac)
                action_sequence = ac
            fin.close()

            print("action number: ", len(action_sequence))
            for act in action_sequence:
                embedding_input.append(self.embedding_input)
                index_input.append(self.index_input)
                
                obs.append(ob[:self.input_shape])
                act = np.array(act)
                acs.append(act)
                
                x.append(ob[0])
                y.append(ob[1])
                z.append(ob[2])
                
                # take that action and record results
                ob, r, done, info = env.step(act)
                # print(ob)
                # ob = np.concatenate((ob[:6], ob[9:12]))
                
                # record result of taking that action
                steps += 1
                next_obs.append(ob[:self.input_shape])
                # rewards.append(r)
                
                # only support rbg_array mode currently
                if render:
                    image = env.get_image(400,400,'leftview')
                    image_obs_left.append(image)
                    image = env.get_image(400,400,'topview')
                    image_obs_front.append(image)
                
        
                success_push_1 = max(success_push_1, info["success_push_1"])
                success_push_2 = max(success_push_2, info["success_push_2"])
                push_dist1 = min(push_dist1, info["pushDist1"])
                push_dist2 = min(push_dist2, info["pushDist2"])
                
                # end the rollout if the rollout ended
                rollout_done = True if (done or steps>=self.max_path_length) else False
                terminals.append(rollout_done)

            print("last observation: ", obs[-1])
        
        # print(x)
        # print(y)
        # print(z)
        
        success = {
            "push_1": success_push_1,
            "push_2": success_push_2
        }
        
        log_info += "expert_success_push_1: " + str(success_push_1) + "\n"
        log_info += "expert_success_push_2: " + str(success_push_2) + "\n"
        log_info += "expert_push_1: " +  str(push_dist1) + "\n"
        log_info += "expert_push_2: " + str(push_dist2) + "\n"
        log_info += "path_length: " + str(len(acs)) + "\n"
        
        
        if render:    
            if len(image_obs_front)>0:
                imageio.mimsave("../Expert/Concurrent/" + tag + "_front.gif", image_obs_front)
            if len(image_obs_left)>0:
                imageio.mimsave("../Expert/Concurrent/" + tag + "_left.gif", image_obs_left)
        
        if log == True:
            print(log_info)
            
                        
        paths = [Path(obs, acs, next_obs, terminals, success, embedding_input, index_input)]
        
        timesteps_this_batch = 0
        for p in paths:
            timesteps_this_batch += len(p["observation"])

        return paths, timesteps_this_batch
    
    
    def run_agent(self, policy, render=False, log = False, log_prefix = "./", n_iter=0, use_embedding=False):
        
        # initialize env for the beginning of a new rollout
        env = self.env_info.env
        env.eval()
        ob = env.reset()
        log_info = "initial ob: " + str(ob) + "\n"
        
        # init vars
        obs, acs, rewards, next_obs, terminals, image_obs_front, image_obs_left, embedding_input, index_input = [], [], [], [], [], [], [], [], []
        steps = 0
        done = False
        success_push_1 = 0
        success_push_2 = 0
        push_dist1 = 0
        push_dist2 = 0
        
        while True:
            embedding_input.append(self.embedding_input)
            index_input.append(self.index_input)
            
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
                image = env.get_image(400,400,'leftview')
                image_obs_left.append(image)
                image = env.get_image(400,400,'frontview')
                image_obs_front.append(image)
            
            success_push_1 = max(success_push_1, info["success_push_1"])
            success_push_2 = max(success_push_2, info["success_push_2"])
            push_dist1 = min(push_dist1, info["pushDist1"])
            push_dist2 = min(push_dist2, info["pushDist2"])

            # end the rollout if the rollout ended
            rollout_done = True if (done or steps>=self.max_path_length) else False
            terminals.append(rollout_done)

            if rollout_done:
                break
        
        log_info += "agent_success_push_1: " + str(success_push_1) + "\n"
        log_info += "agent_success_push_2: " + str(success_push_2) + "\n"
        log_info += "agent_push_1: " +  str(push_dist1) + "\n"
        log_info += "agent_push_2: " + str(push_dist2) + "\n"
        log_info += "path_length: " + str(len(acs)) + "\n"
            
        if len(image_obs_front)>0:
            imageio.mimsave(log_prefix + str(n_iter) + "_agent_front.gif", image_obs_front)
        if len(image_obs_left)>0:
            imageio.mimsave(log_prefix + str(n_iter) + "_agent_left.gif", image_obs_left)
            
        if log == True:
            print(log_info)
        
        success_dict={
            "push_1": success_push_1,
            "push_2": success_push_2
        }
        return success_dict
    
    # the policy gradient should be frozen before sending into this function
    def interface(self, action_file, tag, stdscr):  
        
        render=True
        # initialize env for the beginning of a new rollout
        env = self.env_info.env
        env.eval()
        ob = env.reset()
        # ob = np.concatenate((ob[:6], ob[9:12]))
        ob = np.concatenate((ob[:6], ob[9:12]))
        
        # init vars
        obs, acs, rewards, next_obs, terminals, image_obs_front, image_obs_left, embedding_input, index_input = [], [], [], [], [], [], [], [], []
        steps = 0
        done = False
        
        log_info = ""
        success_push_1 = 0
        success_push_2 = 0
        push_dist1 = 1
        push_dist2 = 1
        
        # read all the actions
        import json
        action_list = []
        action_sequence = []
        
        # for test
        x = []
        y = []
        z = []
        
        index = 0
        
        # do separate steps
        for file in action_file:
            index += 1
            initial_ob = ob
            best_distance = 1
            best_position = None
            print(initial_ob)
            # offset = np.concatenate([initial_ob[:3], np.array([0])]) - np.array([-0.0326475,   0.51488176,  0.23687746, 0])
            # print("offset: ", offset)
            with open(file, 'r') as fin:
                ac = json.load(fin)["actions"]
                action_list.append(ac)
                action_sequence = ac
            fin.close()

            print("action number: ", len(action_sequence))
            for act in action_sequence:
                embedding_input.append(self.embedding_input)
                index_input.append(self.index_input)
                
                obs.append(ob)
                # act = np.array(act) - offset
                act = np.array(act)
                acs.append(act)
                
                x.append(ob[0])
                y.append(ob[1])
                z.append(ob[2])
                
                # take that action and record results
                ob, r, done, info = env.step(act)
                # print(ob)
                ob = np.concatenate((ob[:6], ob[9:12]))
                
                # record result of taking that action
                steps += 1
                next_obs.append(ob)
                rewards.append(r)
                
                # only support rbg_array mode currently
                if render:
                    # image_obs.append(env.render(mode='rgb_array'))
                    image = env.get_image(400,400,'leftview')
                    image_obs_left.append(image)
                    image = env.get_image(400,400,'topview')
                    image_obs_front.append(image)
                
                
                if best_distance > info["pushDist1"]:
                    best_distance = info["pushDist1"]
                    best_position = info["push1"]
                    
                success_push_1 = max(success_push_1, info["success_push_1"])
                success_push_2 = max(success_push_2, info["success_push_2"])
                push_dist1 = min(push_dist1, info["pushDist1"])
                push_dist2 = min(push_dist2, info["pushDist2"])
                
                # success_pick_place = max(success_pick_place, info["success"]["pick_place"])
                
                # end the rollout if the rollout ended
                rollout_done = True if (done or steps>=self.max_path_length) else False
                terminals.append(rollout_done)

            
            if len(image_obs_front)>0:
                # imageio.mimsave(log_prefix + str(n_iter) + "_trial.gif", image_obs)
                imageio.mimsave("../Expert/Concurrent/" + tag + "_front.gif", image_obs_front)

            if len(image_obs_left)>0:
                # imageio.mimsave(log_prefix + str(n_iter) + "_trial.gif", image_obs)
                imageio.mimsave("../Expert/Concurrent/" + tag + "_left.gif", image_obs_left)
            
            print("push1: ", push_dist1, success_push_1)
            print("push2: ", push_dist2, success_push_2) 
                
            import curses
            # human sample 
            # TODO: add more levels that can cover all actions ranging from (-1,1)
            interface_acs = []
            while True:
                keycode = stdscr.getch()
                if keycode == curses.KEY_UP:
                    noise = np.random.randn()/10
                    act = np.array([0, 0, 0.2+noise, 0])
                    print("z Up")
                elif keycode == curses.KEY_DOWN:
                    noise = np.random.randn()/10
                    act = np.array([0, 0, -0.2-noise, 0])
                    print("z Down")
                elif keycode == curses.KEY_LEFT:
                    noise = np.random.randn()/10
                    act = np.array([0, -0.2-noise, 0, 0])
                    print("y left")
                elif keycode == curses.KEY_RIGHT:
                    noise = np.random.randn()/10
                    act = np.array([0, 0.2+noise, 0, 0])
                    print("y right")
                elif keycode == ord('['):
                    noise = np.random.randn()/10
                    act = np.array([-0.3-noise, 0, 0, 0])
                    print("x front")
                elif keycode == ord(']'):
                    noise = np.random.randn()/10
                    act = np.array([0.3+noise, 0, 0, 0])
                    print("x back")
                elif keycode == ord('='):
                    noise = np.random.randn()/5
                    act = np.array([0, 0, 0, -0.6-noise])
                    print("gripper open")
                elif keycode == ord('-'):
                    noise = np.random.randn()/5
                    act = np.array([0, 0, 0, 0.6+noise])
                    print("gripper close")
                elif keycode == ord("a"):
                    noise = np.random.randn()/10
                    act = np.array([0, -0.7-noise, 0, 0])
                    print("y left big")
                elif keycode == ord("d"):
                    noise = np.random.randn()/10
                    act = np.array([0, 0.7+noise, 0, 0])
                    print("y right big")
                elif keycode == ord("w"):
                    noise = np.random.randn()/10
                    act = np.array([0, 0, 0.7+noise, 0])
                    print("z up big")
                elif keycode == ord("s"):
                    noise = np.random.randn()/10
                    act = np.array([0, 0, -0.7-noise, 0])
                    print("z down big")
                elif keycode == ord('o'):
                    act = np.array([0, 0, 0, 0])
                    print("no moves")
                elif keycode == ord('x'):
                    # save observation and action file
                    demonstration = {"actions": interface_acs}
                    demo_json = json.dumps(demonstration, sort_keys=False, indent=4)
                    f = open("../Expert/" + str(index) + ".json", 'w')
                    f.write(demo_json)
                    print(len(interface_acs))
                    break
                else:
                    continue
                    
                # take that action and record results
                ob, r, done, info = env.step(act)
                # print(ob)
                # ob = np.concatenate((ob[:6], ob[9:12]))
                ob = ob[:3]
                interface_acs.append(act.tolist())
                
                # record result of taking that action
                steps += 1
                next_obs.append(ob)
                print(ob)
                rewards.append(r)
                
                # only support rbg_array mode currently
                if render:
                    # image_obs.append(env.render(mode='rgb_array'))
                    image = env.get_image(400,400,'leftview')
                    image_obs_left.append(image)
                    image = env.get_image(400,400,'topview')
                    image_obs_front.append(image)

                    if len(image_obs_front)>0:
                        # imageio.mimsave(log_prefix + str(n_iter) + "_trial.gif", image_obs)
                        imageio.mimsave("../Expert/Concurrent/" + tag + "_front.gif", image_obs_front[steps-10:])
        
                    if len(image_obs_left)>0:
                        # imageio.mimsave(log_prefix + str(n_iter) + "_trial.gif", image_obs)
                        imageio.mimsave("../Expert/Concurrent/" + tag + "_left.gif", image_obs_left[steps-10:])
            
            
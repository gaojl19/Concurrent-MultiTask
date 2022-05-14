from curses import KEY_BACKSPACE
from operator import index
from os import lockf
from re import L
from sys import prefix
from tokenize import Single
import numpy as np
import torch
import time
import copy
import imageio
import json
from utils.utils import *
from metaworld_utils.meta_env import generate_single_mt_env


class ConcurrentCollector():
    def __init__(self, env, env_cls, env_args, env_info, device, max_path_length, min_timesteps_per_batch, input_shape, task_types, embedding_input = [], index_input = None):
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
        self.task_types = task_types
        
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
    def sample_expert(self, action_file, render=True, log=True, plot_prefix=None, tag=None):  
        
        render=True
        log_info = ""
        
        # initialize env for the beginning of a new rollout
        env = self.env_info.env
        env.eval()
        if self.task_types == ["push-2"]:
            ob = env.reset(push_2_init=True)
        else:
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
                imageio.mimsave(plot_prefix + tag + "_expert_front.gif", image_obs_front)
            if len(image_obs_left)>0:
                imageio.mimsave(plot_prefix + tag + "_expert_left.gif", image_obs_left)
        
        if log == True:
            print(log_info)
            
                        
        paths = [Path(obs, acs, next_obs, terminals, success, embedding_input, index_input)]
        
        timesteps_this_batch = 0
        for p in paths:
            timesteps_this_batch += len(p["observation"])

        return paths, timesteps_this_batch
    
    
    def run_agent(self, policy, render=False, log = False, log_prefix = "./", n_iter=0, use_index=False):
        
        # initialize env for the beginning of a new rollout
        success_push_1 = 0
        success_push_2 = 0
        push_dist1 = 1
        push_dist2 = 1
        idx = 0
        success_flag = True
        
        log_info = ""
        
        # test push-1 and push-2 sequentially
        # if only push-1/push-2, test 1
        for task in self.task_types:
            success_1 = 0
            success_2 = 0
            dist_1 = 1
            dist_2 = 1
            
            env = self.env_info.env
            env.eval()
            if task == "push-2":
                ob = env.reset(push_2_init=True)
            else:
                ob = env.reset()
                
            log_info += "initial ob: " + str(ob) + "\n"
            
            # init vars
            obs, acs, rewards, next_obs, terminals, image_obs_front, image_obs_left, embedding_input, index_input = [], [], [], [], [], [], [], [], []
            steps = 0
            done = False
            
            while True:
                embedding_input.append(self.embedding_input)
                index_input.append(self.index_input)
                
                ob = ob[:policy.input_shape]
                obs.append(ob)
                
                if use_index:
                    # print(torch.LongTensor([idx]).reshape(-1, 1))
                    act = policy.get_action(torch.Tensor(ob).to(self.device).unsqueeze(0), torch.LongTensor([idx]).reshape(-1, 1))
                else:
                    act = policy.get_action(torch.Tensor(ob).to(self.device).unsqueeze(0)).detach().cpu().numpy()
                    
                act = np.squeeze(act)
                # log_info += "agent:" + str(act) + "\n"
                    
                acs.append(act)
                
                # take that action and record results
                ob, r, done, info = env.step(act)
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
                
                success_1 = max(success_1, info["success_push_1"])
                success_2 = max(success_2, info["success_push_2"])
                dist_1 = min(dist_1, info["pushDist1"])
                dist_2 = min(dist_2, info["pushDist2"])

                # end the rollout if the rollout ended
                rollout_done = True if (done or steps>=self.max_path_length) else False
                terminals.append(rollout_done)

                if rollout_done:
                    break
            
                
            if len(image_obs_front)>0:
                imageio.mimsave(log_prefix + str(n_iter) + "_" + task + "_agent_front.gif", image_obs_front)
            if len(image_obs_left)>0:
                imageio.mimsave(log_prefix + str(n_iter) + "_" + task +"_agent_left.gif", image_obs_left)
                
            idx += 1
            
            success_push_1 = max(success_push_1, success_1)
            success_push_2 = max(success_push_2, success_2)
            
            # must do the job sequentially
            if (success_2==1 and success_1==1) or (success_2==0 and success_1==0):
                success_flag = False
            
            log_info += "agent_success_push_1: " + str(success_1) + "\n"
            log_info += "agent_success_push_2: " + str(success_2) + "\n"
            log_info += "agent_push_1: " +  str(dist_1) + "\n"
            log_info += "agent_push_2: " + str(dist_2) + "\n"
            log_info += "path_length: " + str(len(acs)) + "\n"  
            log_info += "success_flag: " + str(success_flag) + "\n"
        
        if log == True:
            print(log_info)
        
        success_dict={
            "push_1": success_push_1,
            "push_2": success_push_2,
            "success": success_flag
        }
        return success_dict
    
    
    def keyboar2action(self, stdscr):
        # collect 4 action dimensions
        import curses
        # TODO: add more levels that can cover all actions ranging from (-1,1)
        act = [np.random.randn()/100,np.random.randn()/100,np.random.randn()/100,np.random.randn()/100]
        small_step = 0.2
        medium_step = 0.5
        large_step = 0.8
        cnt = 0
        
        while cnt < 4:
            noise = np.random.randn()/10
            keycode = stdscr.getch()
            
            # small steps
            if keycode == ord("3"):
                act[2] = small_step+noise
                print("z Up small")
            elif keycode == ord("e"):
                act[2] = -small_step-noise
                print("z Down small")
            elif keycode == ord("u"):
                act[1] = -small_step-noise
                print("y left small")
            elif keycode == ord("7"):
                act[1] = small_step+noise
                print("y right small")
            elif keycode == ord('['):
                act[0] = -small_step-0.1-noise
                print("x front small")
            elif keycode == ord(']'):
                act[0] = small_step+0.1+noise
                print("x back small")
                
            # medium steps
            elif keycode == ord('2'):
                act[2] = medium_step+noise
                print("z Up medium")
            elif keycode == ord('w'):
                act[2] = -medium_step-noise
                print("z Down medium")
            elif keycode == ord('y'):
                act[1] = -medium_step-noise
                print("y left medium")
            elif keycode == ord('6'):
                act[1] = medium_step+noise
                print("y right medium")
            elif keycode == ord(';'):
                act[0] = -medium_step-0.1-noise
                print("x front medium")
            elif keycode == ord("'"):
                act[0] = medium_step+0.1+noise
                print("x back medium")
            
            
            # largs steps
            elif keycode == ord("t"):
                act[1] = -large_step-noise*2
                print("y left big")
            elif keycode == ord("5"):
                act[1] = large_step+noise*2
                print("y right big")
            elif keycode == ord("1"):
                act[2] = large_step+noise*2
                print("z up big")
            elif keycode == ord("q"):
                act[2] = -large_step-noise*2
                print("z down big")
            elif keycode == ord(","):
                act[0] = -large_step-1-noise
                print("x front big")
            elif keycode == ord("."):
                act[0] = large_step+1+noise
                print("x back big")
                
                  
            # gripper
            elif keycode == ord('='):
                act[3] = -0.6-noise*3
                print("gripper open")
            elif keycode == ord('-'):
                act[3] = 0.6+noise*3
                print("gripper close")
                
           
            elif keycode == ord(' '):
                pass
                print("no moves")
            elif keycode == ord('x'):
                print("exit interface")
                return np.array([-1,-1,-1,-1])

            elif keycode == ord("p"):
                print("mark")
                return np.array([1,1,1,1])
            
            elif keycode == ord('0'):
                print("reset interface")
                return np.array([0,0,0,0])
            else:
                continue
            
            cnt += 1
        # print(act)
        return np.array(act)
        
    def start_human_demo(self, stdscr, env, prefix):
        
        interface_acs = []
        image_obs_left = []
        image_obs_front = []
        
        sub_traj = []
        
        success_push_1 = 0
        success_push_2 = 0
        push_dist1 = 1
        push_dist2 = 1
        steps = 0
        
        while True:
            act = self.keyboar2action(stdscr)
            if (act == np.array([0,0,0,0])).all():
                return True # reset
                
            elif (act == np.array([-1,-1,-1,-1])).all():
                interface_acs.append(sub_traj)
                
                # save observation and action file
                for i in range(len(interface_acs)):
                    demonstration = {}
                    demonstration["actions"] = interface_acs[i]
                    demo_json = json.dumps(demonstration, sort_keys=False, indent=4)
                    f = open(prefix + "expert_demo_" + str(i+1) + ".json", 'w')
                    f.write(demo_json)
                    f.close()
                print("action length: ", steps, "trajs: ", len(interface_acs))
                
                # replot demo
                if len(image_obs_front)>0:
                    imageio.mimsave(prefix + "expert_front.gif", image_obs_front)
                
                if len(image_obs_left)>0:
                    imageio.mimsave(prefix + "expert_left.gif", image_obs_left)
                    
                return False  # done
            
            elif (act == np.array([1,1,1,1])).all():
                # mark trajectory phase
                interface_acs.append(sub_traj)
                sub_traj = []
                continue
            
            # take that action and record results
            ob, r, done, info = env.step(act)
            # print(ob)
            # ob = np.concatenate((ob[:6], ob[9:12]))
            ob = ob[:3]
            sub_traj.append(act.tolist())
            
            # record result of taking that action
            steps += 1
            print(ob)
            
            # only support rbg_array mode currently
            image = env.get_image(400,400,'leftview')
            image_obs_left.append(image)
            image = env.get_image(400,400,'topview')
            image_obs_front.append(image)

            print(len(image_obs_front))
            if len(image_obs_front)>0:
                # imageio.mimsave(log_prefix + str(n_iter) + "_trial.gif", image_obs)
                if len(image_obs_front)<10:
                    imageio.mimsave(prefix + "expert_front.gif", image_obs_front)
                else: 
                    imageio.mimsave(prefix + "expert_front.gif", image_obs_front[len(image_obs_front)-10:])

            if len(image_obs_left)>0:
                # imageio.mimsave(log_prefix + str(n_iter) + "_trial.gif", image_obs)
                if len(image_obs_left)<10:
                    imageio.mimsave(prefix + "expert_left.gif", image_obs_left)
                else: 
                    imageio.mimsave(prefix + "expert_left.gif", image_obs_left[len(image_obs_left)-10:])
            
            success_push_1 = max(success_push_1, info["success_push_1"])
            success_push_2 = max(success_push_2, info["success_push_2"])
            push_dist1 = min(push_dist1, info["pushDist1"])
            push_dist2 = min(push_dist2, info["pushDist2"])
            
            print("push1: ", push_dist1, success_push_1)
            print("push2: ", push_dist2, success_push_2) 
            
                        
    # the policy gradient should be frozen before sending into this function
    def interface(self, action_file, tag, stdscr, prefix):  
        
        render=True
        # initialize env for the beginning of a new rollout
        env = self.env_info.env
        env.eval()
        ob = env.reset()
        # ob = np.concatenate((ob[:6], ob[9:12]))
        ob = np.concatenate((ob[:6], ob[9:12]))
        
        
        if len(action_file) == 0:
            print("collecting interface!")
            reset = True
            while reset:
                reset = self.start_human_demo(stdscr, env, prefix)
        
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
            
            self.start_human_demo(stdscr, env, prefix)
            
            
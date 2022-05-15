import torch.nn as nn
import numpy as np
import random
import torch
from torch.distributions import Normal

from .distribution import TanhNormal
from shutil import Error
from .continuous_policy import MultiHeadGuassianContPolicy

EPSILON = 1e-20

class EMMultiHeadPolicy(nn.Module):
    def __init__(self, input_shape, output_shape, head_num, params):
        super().__init__()
        self.policy = MultiHeadGuassianContPolicy (
            input_shape = input_shape, 
            output_shape = 2 * output_shape,
            head_num=head_num,
            **params['net'] 
        )
        self.input_shape = input_shape
        self.head_num = head_num
        self.last_weights = torch.Tensor()
        # print(env.observation_space.shape[0])
    
    
    
    def train_forward(self, obs, acs, idx, criterion, log):
        '''
            1. calculate latent variable z, detach
            2. calculate loss, using reparameterization trick
        '''
        
        # 1. calculate latent variable z(weights), detach
        mean_list, std_list, log_std_list = self.policy.train_forward(obs)
        prob = []
        if log:
            print("mean:", mean_list)
            print("std: ", std_list)
            print("actions: ", acs)
            
        for i in range(len(mean_list)):
            mean = mean_list[i]
            std = std_list[i]
            log_std = log_std_list[i]
            
            dis = Normal(mean, std)
            # print(acs.shape)
            log_prob = dis.log_prob(acs)
            if log:
                print("log probability: ", log_prob)
            
            log_prob = log_prob.sum(dim=-1)
            # print(log_prob)
            # print(torch.exp(log_prob))
            
            prob.append(log_prob)

        prob = torch.stack(prob).detach()
        
        # 2. normalize and calculate weighted loss
        weights = []
        
        for i in range(self.head_num):
            diff = 0
            for j in range(self.head_num):
                diff_ = prob[j] - prob[i]
                diff += torch.exp(diff_)
            weights.append(1/diff)
            
        weights = torch.stack(weights)
        weights = weights/weights.sum(dim=0)
        if log:
            print("weights: ", weights)
        
        loss = []
        for i in range(len(mean_list)):
            mean = mean_list[i]
            std = std_list[i]
            
            dis = TanhNormal(mean, std)
            action = dis.rsample( return_pretanh_value = False)
            loss.append(criterion(action, acs).sum(dim=1)/acs.shape[1])
            
        loss = torch.stack(loss).reshape(weights.shape)
        loss = loss*weights
        
        loss = loss.sum(dim=0) # average over multi-heads
        loss = loss.sum(dim=0)/loss.shape[0] # average over batch samples
        
        # print weights differences
        if log:
            if self.last_weights.shape == weights.shape:
                print("weights difference with last: ", abs(self.last_weights-weights).sum(dim=0), abs(self.last_weights-weights).sum(dim=0).sum(dim=0))
            weight_idx = weights.argmax(dim=0)
            idx = idx.reshape(weight_idx.shape)
            # print(weight_idx)
            # print(idx)
            print("weights differences with truth: ", abs(weight_idx-idx), abs(weight_idx-idx).sum(dim=0))
            # exit(0)
        self.last_weights = weights
        
        return loss

    
    def eval_act(self, obs: np.ndarray, idx):
        with torch.no_grad():
            mean, std, log_std = self.policy.forward(obs, idx)
        return mean.squeeze(0).detach().cpu().numpy()
    
    def get_action(self, obs: np.ndarray, idx):
        # if len(obs.shape) > 1:
        #     observation = obs
        # else:
        #     observation = obs[None]

        # # TODO return the action that the policy prescribes
        # return self.forward(observation, idx)
        return self.eval_act(obs, idx)


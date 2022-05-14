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
        # print(env.observation_space.shape[0])
    
    
    
    def train_forward(self, obs, acs, criterion):
        '''
            1. calculate latent variable z, detach
            2. calculate loss, using reparameterization trick
        '''
        
        # 1. calculate latent variable z(weights), detach
        mean_list, std_list, log_std_list = self.policy.train_forward(obs)
        prob = []
        for i in range(len(mean_list)):
            mean = mean_list[i]
            std = std_list[i]
            log_std = log_std_list[i]
            
            dis = Normal(mean, std)
            # print(acs.shape)
            log_prob = dis.log_prob(acs)
            print("actions: ", acs)
            print("log probability: ", log_prob)
            log_prob = log_prob.sum(dim=-1)
            # print(log_prob)
            # print(torch.exp(log_prob))
            
            prob.append(log_prob)

        prob = torch.stack(prob).detach()
        
        # 2. normalize and calculate weighted loss
        # weights += EPSILON # in case of both zero probability
        # weights = weights/weights.sum(dim=0)
        weights = []
        
        # only for n=2
        for i in range(2):
            diff = prob.sum(dim=0) - prob[i]*2
            weights.append(1/(torch.exp(diff)+1))
        # weights = torch.stack(weights)
        # print("original weights: ", weights)
        # exit(0)
        
        weights = torch.stack(weights)
        print(weights)
        weights = weights/weights.sum(dim=0)
        
        loss = []
        for i in range(len(mean_list)):
            mean = mean_list[i]
            std = std_list[i]
            
            dis = TanhNormal(mean, std)
            action = dis.rsample( return_pretanh_value = False)
            # action = mean
            loss.append(criterion(action, acs).sum(dim=1)/acs.shape[1])
            # print("loss: ", criterion(action, acs).sum(dim=1).sum(dim=0)/32/4)
            
        loss = torch.stack(loss).reshape(weights.shape)
        # print(loss)
        loss = loss*weights
        # print(loss.shape)
        # print(weights.shape)
        
        loss = loss.sum(dim=0) # average over multi-heads
        loss = loss.sum(dim=0)/loss.shape[0] # average over batch samples
            
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


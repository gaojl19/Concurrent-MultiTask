import torch.nn as nn
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .distribution import TanhNormal
from .continuous_policy import MultiHeadGuassianContPolicy
import networks.init as init

EPSILON = 1e-20
class Encoder(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shape, head_num):
        super().__init__()
        
        self.activation_func = F.relu
        self.encoder = []
        
        input_shape = input_shape + output_shape
        for i in range(2):
            fc = nn.Linear(input_shape, hidden_shape)
            input_shape = hidden_shape
            init.basic_init(fc)
            self.encoder.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("encoder{}".format(i), fc)
        self.last_activation_func = nn.Softmax()
        
        
        # encoder output: [hidden_shape, output_shape]
        fc = nn.Linear(hidden_shape, head_num)
        init.basic_init(fc)
        self.encoder.append(fc)
        self.__setattr__("encoder{}".format(i), fc)
    
    
    def forward(self, obs, acs):
        # print(obs.shape)
        # print(acs.shape)
        out = torch.cat((obs, acs), dim=1)
        # print(out.shape)
        
        for fc in self.encoder[:-1]:
            out = fc(out)
            out = self.activation_func(out)
            
        out = self.encoder[-1](out)
        
        if self.last_activation_func != None:
            out = self.last_activation_func(out)
        
        # print("out: ", out)
        return out
    
class VAEMultiHeadPolicy(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shape, head_num, params, soft):
        super().__init__()
        
        self.encoder = Encoder(
            input_shape=input_shape, 
            output_shape=output_shape,
            hidden_shape=hidden_shape, 
            head_num=head_num
        )
        
            
        self.policy = MultiHeadGuassianContPolicy (
            input_shape = input_shape, 
            output_shape = 2 * output_shape,
            head_num=head_num,
            **params['net'] 
        )
        self.input_shape = input_shape
        self.head_num = head_num
        self.last_weights = torch.Tensor()
        self.soft = soft
        # print(env.observation_space.shape[0])
    
    
    def train_forward(self, obs, acs, idx, criterion, log):
        '''
            1. calculate latent variable z(probability of )
            2. calculate loss, using reparameterization trick
        '''
        
        # 1. calculate latent variable z(weights)
        if self.soft:
            weights = self.encoder(obs,acs)
            
            if log:
                print("weights: ", weights)
            
            # 2. calculate different loss
            mean_list, std_list, log_std_list = self.policy.train_forward(obs)
            loss = []
            for i in range(len(mean_list)):
                mean = mean_list[i]
                std = std_list[i]
                
                dis = TanhNormal(mean, std)
                action = dis.rsample( return_pretanh_value = False)
                loss.append(criterion(action, acs).sum(dim=1)/acs.shape[1])  
            
            # 3. average loss over weights
            loss = torch.stack(loss).reshape(weights.shape)
            # print(loss.shape, weights.shape)
            loss = loss*weights
            
            loss = loss.sum(dim=1) # average over multi-heads
            loss = loss.sum(dim=0)/loss.shape[0] # average over batch samples
            
            # print weights differences
            if log:
                if self.last_weights.shape == weights.shape:
                    print("weights difference with last: ", abs(self.last_weights-weights).sum(dim=1), abs(self.last_weights-weights).sum(dim=1).sum(dim=0))
                weight_idx = weights.argmax(dim=1)
                idx = idx.reshape(weight_idx.shape)

                print("weights differences with truth: ", abs(weight_idx-idx), abs(weight_idx-idx).sum(dim=0))
            self.last_weights = weights
        
        
        ## Hard update
        else:
            weights = self.encoder(obs,acs)
            # print(weights)
    
            idx = weights.argmax(dim=1, keepdim=True)
            if log:
                print(idx.reshape(1,-1))
            
            mean, std, log_std = self.policy.forward(obs, idx)
            dis = TanhNormal(mean, std)
            action = dis.rsample( return_pretanh_value = False )
        
            loss = criterion(action, acs)
        
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


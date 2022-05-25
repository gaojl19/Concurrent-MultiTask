from requests import head
import torch.nn as nn
import numpy as np
import random
import torch


from .distribution import TanhNormal
from shutil import Error
from .continuous_policy import SingleHeadContPolicy

class MultiNetPolicy(nn.Module):
    def __init__(self, input_shape, output_shape, head_num, params):
        super().__init__()
        self.policy = MultiNet(input_shape=input_shape,
                               output_shape=output_shape,
                               head_num=head_num,
                               params=params)
        self.input_shape = input_shape
    
    def train_forward(self, obs, acs, criterion, log):
        return self.policy.train_forward(obs, acs, criterion, log)
        
    def get_action(self, obs: np.ndarray, idx):
        return self.policy.get_action(obs, idx)
    
    
class MultiNet(nn.Module):
    def __init__(self, input_shape, output_shape, head_num, params, beta=0.01):
        super().__init__()
        self.networks = []
        for _ in range(head_num):
            policy = SingleHeadContPolicy(
                input_shape = input_shape,
                output_shape = 2 * output_shape,
                n_layers = params['n_layers'],
                hidden_shape = params['size']
            )
            self.networks.append(policy)
            self.__setattr__("network{}".format(_), policy)
        
        self.input_shape = input_shape
        self.head_num = head_num
        self.beta = beta
        # print(env.observation_space.shape[0])
        
    
    def train_forward(self, obs, acs, criterion, log):
        
        mean_list = []
        action_list = []
        for i in range(self.head_num):
            mean, std, log_std = self.networks[i].forward(obs)
            dis = TanhNormal(mean, std)
            action = dis.rsample( return_pretanh_value = False )
            
            mean_list.append(mean)
            action_list.append(action)
        
        # prediction loss
        loss1 = 0
        for action in action_list:
            loss1 += abs(action-acs).mean()
        loss1 /= len(action_list)
        
        # sparsity loss
        loss2 = 0
        cnt = 0
        for i in range(self.head_num):
            for j in range(i+1, self.head_num):
                cnt += 1
                loss2 += abs(mean_list[i] - mean_list[j]).mean()
        loss2 /= cnt
        
        loss = loss1 + 1/loss2*self.beta
        
        return loss
    
    def eval_act(self, obs: np.ndarray, idx):
        assert idx.shape[0] == 1  # single input
        
        with torch.no_grad():
            mean, std, log_std = self.networks[idx[0]].forward(obs)
            
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()
    
    def get_action(self, obs: np.ndarray, idx):
        # if len(obs.shape) > 1:
        #     observation = obs
        # else:
        #     observation = obs[None]

        # # TODO return the action that the policy prescribes
        # return self.forward(observation, idx)
        return self.eval_act(obs, idx)


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
        self.networks = []
        for _ in range(head_num):
            self.networks.append(SingleHeadContPolicy(
                input_shape = input_shape,
                output_shape = 2 * output_shape,
                n_layers = self.agent_params['n_layers'],
                hidden_shape = self.agent_params['size']
            ))
        
        self.input_shape = input_shape
        self.head_num = head_num
        # print(env.observation_space.shape[0])
        
    
    def train_forward(self, obs, acs, criterion, log):
        
        for i in range(self.head_num):
            mean, std, log_std = self.networks[i].forward(obs)
            dis = TanhNormal(mean, std)
            action = dis.rsample( return_pretanh_value = False )
            
            
        return action
    
    def eval_act(self, obs: np.ndarray, idx):
        with torch.no_grad():
            mean, std, log_std = self.policy.forward(obs, idx)
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()
    
    def get_action(self, obs: np.ndarray, idx):
        # if len(obs.shape) > 1:
        #     observation = obs
        # else:
        #     observation = obs[None]

        # # TODO return the action that the policy prescribes
        # return self.forward(observation, idx)
        return self.eval_act(obs, idx)


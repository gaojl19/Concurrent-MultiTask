import torch.nn as nn
import numpy as np
import random
import torch


from .distribution import TanhNormal
from shutil import Error
from .continuous_policy import MultiHeadGuassianContPolicy

class MultiHeadPolicy(nn.Module):
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
        
    
    def forward(self, obs, idx=None):
        if idx == None:
            idx = []
            for i in range(obs.shape[0]):
                idx.append(np.array(random.sample(range(self.head_num), 1)))
            idx = torch.LongTensor(idx).reshape(-1, 1)
            # print(idx)
            
        mean, std, log_std = self.policy.forward(obs, idx)
        dis = TanhNormal(mean, std)
        action = dis.rsample( return_pretanh_value = False )
        return action
    
    
    def get_action(self, obs: np.ndarray, idx):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        return self.forward(observation, idx)


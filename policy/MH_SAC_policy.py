import torch.nn as nn
import numpy as np

from .distribution import TanhNormal
from shutil import Error
from .continuous_policy import MultiHeadGuassianContPolicy

class MHSACPolicy(nn.Module):
    def __init__(self, env, params):
        super().__init__()
        self.policy = MultiHeadGuassianContPolicy (
            input_shape = env.observation_space.shape[0], 
            output_shape = 2 * env.action_space.shape[0],
            head_num=env.num_tasks,
            **params['net'] 
        )
        self.input_shape = env.observation_space.shape[0]
        # print(env.observation_space.shape[0])
        
    
    def forward(self, obs, idx=None):
        if idx == None:
            raise Error
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


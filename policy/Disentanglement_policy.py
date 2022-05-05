import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networks.init as init

from .distribution import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class DisentanglementPolicy(nn.Module):
    def __init__(self, state_shape, output_shape, hidden_shape, example_embedding, activation_func=F.relu, init_func = init.basic_init, last_activation_func = None ):
        super().__init__()
        
        self.activation_func = activation_func
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = None
    
        
        # state embedding
        # self.embedding1 = nn.Linear(state_shape, hidden_shape)
        # self.embedding2 = nn.Linear(example_embedding.shape[0], hidden_shape)
        # init_func(self.embedding1)
        # init_func(self.embedding2)
        # self.__setattr__("embedding1", self.embedding1)
        # self.__setattr__("embedding2", self.embedding2)
        self.input_shape = state_shape + example_embedding.shape[0]
        
        self.embedding = nn.Linear(state_shape+example_embedding.shape[0], hidden_shape)
        init_func(self.embedding)
        self.__setattr__("embedding", self.embedding)
        
        
        # encoder
        fc = nn.Linear(hidden_shape, example_embedding.shape[0])
        init_func(fc)
        self.encoder = fc
        # set attr for pytorch to track parameters( device )
        self.__setattr__("encoder", fc)
        
        # decoder
        fc = nn.Linear(example_embedding.shape[0], output_shape)
        init_func(fc)
        self.decoder = fc
        self.__setattr__("decoder", fc)
        
        
    
    def forward(self, x, return_feature = True):

        input = self.activation_func(self.embedding(x))
        
        feature = self.activation_func(self.encoder(input))
        out = self.decoder(feature)
        
        if self.last_activation_func!=None:
            out = self.last_activation_func(out)
        
        if return_feature:
            return out, feature
        else:
            return out
    
    
    def get_action(self, obs: np.ndarray):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # return the action that the policy prescribes
        return self.forward(observation, return_feature=False)



class DisentangleMultiHeadPolicy(nn.Module):
    def __init__(self, state_shape, output_shape, hidden_shape, example_embedding, activation_func=F.relu, init_func = init.basic_init, last_activation_func = None ):
        super().__init__()
        
        self.activation_func = activation_func
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = None

        print(example_embedding)
        self.input_shape = state_shape + example_embedding.shape[0]
        self.state_dim = state_shape
        self.embed_dim = example_embedding.shape[0]
        self.act_dim = output_shape
    
        
        self.embedding = nn.Linear(state_shape+example_embedding.shape[0], hidden_shape)
        init_func(self.embedding)
        self.__setattr__("embedding", self.embedding)
        
        
        # encoder
        fc = nn.Linear(hidden_shape, example_embedding.shape[0])
        init_func(fc)
        self.encoder = fc
        # set attr for pytorch to track parameters( device )
        self.__setattr__("encoder", fc)
        
        # decoder
        fc = nn.Linear(example_embedding.shape[0], output_shape * 2 * example_embedding.shape[0])
        init_func(fc)
        self.decoder = fc
        self.__setattr__("decoder", fc)
        
        
    
    def forward(self, x, return_feature = True):

        input = self.activation_func(self.embedding(x))
        
        feature = self.activation_func(self.encoder(input))
        out = self.decoder(feature)
        
        if self.last_activation_func!= None:
            out = self.last_activation_func(out)
        
        # get mean and std from the output
        task_embedding = x[:, self.state_dim:]
        
        mask = torch.arange(0, task_embedding.shape[1], 1).reshape(task_embedding.shape[1], 1).float()
        label = torch.mm(task_embedding, mask).long()
        label = label.reshape(label.shape[0], 1, 1)
        label = label.repeat(1, 1, self.act_dim*2)
       
        out = out.reshape(out.shape[0], -1, self.act_dim*2)
        out = out.gather(1, label)
        
        mean, log_std= out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)

        # calculate standard deviation
        std = torch.exp(log_std)
        dis = TanhNormal(mean, std)
        action = dis.rsample( return_pretanh_value = False )
        
        if return_feature:
            return action, feature
        else:
            return action
    
    
    def get_action(self, obs: np.ndarray):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # return the action that the policy prescribes
        return self.forward(observation, return_feature=False)
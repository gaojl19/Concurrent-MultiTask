import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networks.init as init

class MLPPolicy(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shape, n_layers, activation_func=F.relu, init_func = init.basic_init, last_activation_func = None ):
        super().__init__()
        
        self.activation_func = activation_func
        self.fcs = []
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = None
        input_shape = np.prod(input_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        for i in range(n_layers):
            fc = nn.Linear(input_shape, hidden_shape)
            input_shape = hidden_shape
            init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("fc{}".format(i), fc)
        
        
        # last layer: [hidden_shape, output_shape]
        fc = nn.Linear(hidden_shape, output_shape)
        init_func(fc)
        self.fcs.append(fc)
        self.__setattr__("fc{}".format(i), fc)
        
        
    
    def forward(self, x):
        out = x
        for fc in self.fcs[:-1]:
            out = fc(out)
            out = self.activation_func(out)
        out = self.fcs[-1](out)
        
        if self.last_activation_func != None:
            out = self.last_activation_func(out)
        return out
    
    
    def get_action(self, obs: np.ndarray):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # return the action that the policy prescribes
        return self.forward(observation)

    
    
    
    # def update(
    #         self, observations, actions,
    #         adv_n=None, acs_labels_na=None, qvals=None):
        
    #     # TODO: update the policy and return the loss
    #     pred_acs = self.forward(observations)
    #     loss = self.loss(pred_acs, actions)
        
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
        
    #     return {
    #         # You can add extra logging information here, but keep this line
    #         'Training Loss': loss.to('cpu').detach().numpy(),
    #     }

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import networks as networks
from .distribution import TanhNormal
import torch.nn.functional as F
import networks.init as init

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class UniformPolicyContinuous(nn.Module):
    def __init__(self, action_shape):
        super().__init__()
        self.action_shape = action_shape

    def forward(self, x):
        return torch.Tensor(np.random.uniform(-1., 1., self.action_shape))

    def explore(self, x):
        return {
            "action": torch.Tensor(
                np.random.uniform(-1., 1., self.action_shape))
        }


class DetContPolicy(networks.Net):
    def forward(self, x):
        return torch.tanh(super().forward(x))

    def eval_act( self, x ):
        with torch.no_grad():
            return self.forward(x).squeeze(0).detach().cpu().numpy()

    def explore( self, x ):
        return {
            "action": self.forward(x).squeeze(0)
        }


class FixGuassianContPolicy(networks.Net):
    def __init__(self, norm_std_explore, **kwargs):
        super().__init__(**kwargs)
        self.norm_std_explore = norm_std_explore

    def forward(self, x):
        return torch.tanh(super().forward(x))

    def eval_act(self, x):
        with torch.no_grad():
            return self.forward(x).squeeze(0).detach().cpu().numpy()

    def explore(self, x):
        action = self.forward(x).squeeze(0)
        action += Normal(
                torch.zeros(action.size()),
                self.norm_std_explore * torch.ones(action.size())
        ).sample().to(action.device)

        return {
            "action": action
        }


class GuassianContPolicy(networks.Net):
    def forward(self, x):
        x = super().forward(x)

        mean, log_std = x.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        return mean, std, log_std

    def eval_act(self, x):
        with torch.no_grad():
            mean, _, _ = self.forward(x)
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()

    def explore( self, x, return_log_probs = False, return_pre_tanh = False ):

        mean, std, log_std = self.forward(x)

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True) 

        dic = {
            "mean": mean,
            "log_std": log_std,
            "ent":ent
        }

        if return_log_probs:
            action, z = dis.rsample(return_pretanh_value=True)
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample(return_pretanh_value=True)
                dic["pre_tanh"] = z.squeeze(0)
            action = dis.rsample(return_pretanh_value=False)

        dic["action"] = action.squeeze(0)
        return dic

    def update(self, obs, actions):
        mean, std, log_std = self.forward(obs)
        dis = TanhNormal(mean, std)

        log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
        ent = dis.entropy().sum(-1, keepdim=True) 
        
        out = {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "ent": ent
        }
        return out


class GuassianContPolicyBasicBias(networks.Net):

    def __init__(self, output_shape, **kwargs):
        super().__init__(output_shape=output_shape, **kwargs)
        self.logstd = nn.Parameter(torch.zeros(output_shape))

    def forward(self, x):
        mean = super().forward(x)

        logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(logstd)
        std = std.unsqueeze(0).expand_as(mean)
        return mean, std, logstd
    
    def eval_act( self, x ):
        with torch.no_grad():
            mean, std, log_std = self.forward(x)
        # return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()
        return mean.squeeze(0).detach().cpu().numpy()
    
    def explore(self, x, return_log_probs = False, return_pre_tanh = False):

        mean, std, log_std = self.forward(x)

        dis = Normal(mean, std)
        # dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(1, keepdim=True) 
        
        dic = {
            "mean": mean,
            "log_std": log_std,
            "ent": ent
        }

        if return_log_probs:
            action = dis.sample()
            log_prob = dis.log_prob(action)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            # dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            # if return_pre_tanh:
                # action, z = dis.rsample(return_pretanh_value=True)
                # dic["pre_tanh"] = z.squeeze(0)
            action = dis.sample()

        dic["action"] = action.squeeze(0)
        return dic

        # if return_log_probs:
        #     action, z = dis.rsample(return_pretanh_value=True)
        #     log_prob = dis.log_prob(
        #         action,
        #         pre_tanh_value=z
        #     )
        #     log_prob = log_prob.sum(dim=1, keepdim=True)
        #     dic["pre_tanh"] = z.squeeze(0)
        #     dic["log_prob"] = log_prob
        # else:
        #     if return_pre_tanh:
        #         action, z = dis.rsample(return_pretanh_value=True)
        #         dic["pre_tanh"] = z.squeeze(0)
        #     action = dis.rsample(return_pretanh_value=False)

        # dic["action"] = action.squeeze(0)
        # return dic

    def update(self, obs, actions):
        mean, std, log_std = self.forward(obs)
        # dis = TanhNormal(mean, std)
        dis = Normal(mean, std)
        # dis = TanhNormal(mean, std)

        log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
        ent = dis.entropy().sum(1, keepdim=True) 
        
        out = {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "ent": ent
        }
        return out

class EmbeddingGuassianContPolicyBase:

    def eval_act( self, x, embedding_input ):
        with torch.no_grad():
            mean, std, log_std = self.forward(x, embedding_input)
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()
    
    def explore( self, x, embedding_input, return_log_probs = False, return_pre_tanh = False ):
        
        mean, std, log_std = self.forward(x, embedding_input)

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True) 
        
        dic = {
            "mean": mean,
            "log_std": log_std,
            "ent":ent
        }

        if return_log_probs:
            action, z = dis.rsample( return_pretanh_value = True )
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample( return_pretanh_value = True )
                dic["pre_tanh"] = z.squeeze(0)
            action = dis.rsample( return_pretanh_value = False )

        dic["action"] = action.squeeze(0)
        return dic

    def update(self, obs, embedding_input, actions):
        mean, std, log_std = self.forward(obs, embedding_input)
        dis = TanhNormal(mean, std)

        log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
        ent = dis.entropy().sum(1, keepdim=True) 
        
        out = {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "ent": ent
        }
        return out


class EmbeddingDetContPolicyBase:
    def eval_act( self, x, embedding_input ):
        with torch.no_grad():
            return torch.tanh(self.forward(x, embedding_input)).squeeze(0).detach().cpu().numpy()


    def explore( self, x, embedding_input ):
        return {
            "action":torch.tanh(
                self.forward(x, embedding_input)).squeeze(0)}


class ModularGuassianGatedCascadeCondContPolicy(networks.ModularGatedCascadeCondNet, EmbeddingGuassianContPolicyBase):
    def forward(self, x, embedding_input, return_weights = False, dropout = False):
        x = super().forward(x, embedding_input, return_weights = return_weights, dropout = dropout)
        if isinstance(x, tuple):
            general_weights = x[1]
            last_weights = x[2]
            x = x[0]

        mean, log_std = x.chunk(2, dim=-1)
        # print("x shape:", x.shape)
        # print("mean shape:", mean.shape)
        # print("embedding shape: ", embedding_input.shape)

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        if return_weights:
            # return mean, std, log_std, general_weights, last_weights
            return mean, std, log_std, general_weights
        return mean, std, log_std

    def eval_act( self, x, embedding_input, return_weights = False ):
        with torch.no_grad():
            if return_weights:
                # mean, std, log_std, general_weights, last_weights = self.forward(x, embedding_input, return_weights)
                mean, std, log_std, general_weights = self.forward(x, embedding_input, return_weights)
            else:
                mean, std, log_std = self.forward(x, embedding_input, return_weights)
        if return_weights:
            # return torch.tanh(mean.squeeze(0)).detach().cpu().numpy(), general_weights, last_weights
            return torch.tanh(mean.squeeze(0)).detach().cpu().numpy(), general_weights
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()


    def explore( self, x, embedding_input, return_log_probs = False,
                    return_pre_tanh = False, return_weights = False ):
        if return_weights:
            mean, std, log_std,  general_weights, last_weights = self.forward(x, embedding_input, return_weights)
            # general_weights, last_weights = weights
            dic = {
                "general_weights": general_weights,
                "last_weights": last_weights
            }
        else:
            mean, std, log_std = self.forward(x, embedding_input)
            dic = {}

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True) 
        
        dic.update({
            "mean": mean,
            "log_std": log_std,
            "ent":ent
        })

        if return_log_probs:
            action, z = dis.rsample( return_pretanh_value = True )
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample( return_pretanh_value = True )
                dic["pre_tanh"] = z.squeeze(0)
            action = dis.rsample( return_pretanh_value = False )

        dic["action"] = action.squeeze(0)
        return dic


class MultiHeadGuassianContPolicy(networks.BootstrappedNet):
    def train_forward(self, x):
        dist_out = super().train_forward(x)
        mean_list = []
        std_list = []
        log_std_list = []
        
        for x in dist_out:
            mean, log_std = x.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            
            # calculate standard deviation
            std = torch.exp(log_std)
            mean_list.append(mean)
            log_std_list.append(log_std)
            std_list.append(std)
            
            # print(mean, std)
        return mean_list, std_list, log_std_list
        
        
    def forward(self, x, idx):
        
        x = super().forward(x, idx)
        # divide output into 2 parts: mean and covariance
        mean, log_std = x.chunk(2, dim=-1)
        
        # print("mean shape: ", mean.shape) [1,4]
        # print("log_std shape: ", log_std.shape) [1,4]

        # make sure that min <= log_std <= max
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        
        # calculate standard deviation
        std = torch.exp(log_std)

        return mean, std, log_std

    def eval_act( self, x, idx ):
        with torch.no_grad():
            mean, _, _= self.forward(x, idx)
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()

    def random_act(self, x, embedding_input):
        with torch.no_grad():
            mean, std, log_std = self.forward(x, embedding_input)
        
        dis = TanhNormal(mean, std)
        action = dis.rsample( return_pretanh_value = False )

        return action.squeeze(0).detach().cpu().numpy()

    def explore( self, x, idx, return_log_probs=False, return_pre_tanh=False):
        mean, std, log_std = self.forward(x, idx)

        dis = TanhNormal(mean, std)

        # print(dis.entropy().shape) [1,4] -- same as mean and log_std
        # sum all dimensions
        ent = dis.entropy().sum(-1, keepdim=True) 

        dic = {
            "mean": mean,
            "log_std": log_std,
            "ent":ent
        }

        if return_log_probs:
            # sample z from N(mean, std)
            # action = tanh(z) [batch_size, 1, 4]  z = z [batch_size, 1, 4]
            action, z = dis.rsample( return_pretanh_value = True )
            
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            
            # ??
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample( return_pretanh_value = True )
                dic["pre_tanh"] = z.squeeze(0)
            action = dis.rsample( return_pretanh_value = False )

        dic["action"] = action.squeeze(0)
        return dic
    

class SingleHeadContPolicy(nn.Module):
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
        
        
        # last layer: output mean and variance
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
        
        mean, log_std = out.chunk(2, dim=-1)
        # make sure that min <= log_std <= max
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        
        # calculate standard deviation
        std = torch.exp(log_std)
        
        return mean, std, log_std
        

    def eval_act( self, x ):
        with torch.no_grad():
            mean, _, _ = self.forward(x)
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()

    def get_action(self, x):
        return self.eval_act(x)

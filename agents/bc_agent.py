from statistics import variance
from torch_rl.replay_buffer import ReplayBuffer
from policy.MLP_policy import MLPPolicy
from policy.MultiHead_policy import MultiHeadPolicy
from policy.EM_policy import EMMultiHeadPolicy
from policy.VAE_policy import *
from .base_agent import BaseAgent
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
# import hydra


class MLPAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MLPAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicy(
            input_shape = self.agent_params['ob_dim'],
            output_shape = self.agent_params['ac_dim'],
            n_layers = self.agent_params['n_layers'],
            hidden_shape = self.agent_params['size']
        )
        
        print("actor: \n", self.actor)

        # update
        self.loss = nn.MSELoss()
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na):
        self.actor.train()
        
        self.optimizer.zero_grad()
        
        pred_acs = self.actor(ob_no)
        loss = self.loss(pred_acs, ac_na)

        loss.backward()
        self.optimizer.step()
        
        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        return log

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)
        
    def add_mt_to_replay_buffer(self, paths):
        self.replay_buffer.add_mt_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size) 

    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_random_data_embedding(batch_size) 

    def save(self, path):
        return self.actor.save(path)
    
    
    
class MLPEmbeddingAgent(BaseAgent):
    def __init__(self, env, example_embedding, agent_params, params):
        super(MLPEmbeddingAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        # add the embedding vector to the input
        self.actor = MLPPolicy(
            input_shape = self.agent_params['ob_dim'] + np.prod(example_embedding.shape),
            output_shape = self.agent_params['ac_dim'],
            n_layers = self.agent_params['n_layers'],
            hidden_shape = self.agent_params['size']
        )
        
        print("actor: \n", self.actor)

        # update
        self.loss = nn.MSELoss()
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, embedding_input_n, alternate=-1):
        self.actor.train()
        
        self.optimizer.zero_grad()
        input = torch.Tensor(np.concatenate((ob_no, embedding_input_n.squeeze()), axis=1))
        
        pred_acs = self.actor(input)
        loss = self.loss(pred_acs, ac_na)

        loss.backward()
        self.optimizer.step()
        
        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        
        return log

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)
        
    def add_mt_to_replay_buffer(self, paths):
        self.replay_buffer.add_embedding_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size) 

    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_random_data_embedding(batch_size) 

    def save(self, path):
        return self.actor.save(path)

    
class MultiHeadAgent(BaseAgent):
    def __init__(self, env, agent_params, params):
        super(MultiHeadAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MultiHeadPolicy(
            input_shape = self.agent_params['ob_dim'],
            output_shape = self.agent_params['ac_dim'],
            head_num = self.agent_params["head_num"],
            params = params
        )
        
        print("actor: \n", self.actor.policy)
        
        # update
        self.loss = nn.MSELoss()
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na):
        self.actor.train()
        
        self.optimizer.zero_grad()
        pred_acs = self.actor(ob_no, None)
        loss = self.loss(pred_acs, ac_na)
        # print(loss, loss.shape)
        loss.backward()
        self.optimizer.step()
        
        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        return log

    def add_mt_to_replay_buffer(self, paths):
        self.replay_buffer.add_index_rollouts(paths)
    
    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size, complete=False):
        print("complete: ", complete)
        return self.replay_buffer.sample_random_data(batch_size, complete=complete) 
    
    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_random_data_index(batch_size)

    def save(self, path):
        return self.actor.save(path)
    
 

class EMMultiHeadAgent(BaseAgent):
    def __init__(self, env, agent_params, params):
        super(EMMultiHeadAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = EMMultiHeadPolicy(
            input_shape = self.agent_params['ob_dim'],
            output_shape = self.agent_params['ac_dim'],
            head_num = self.agent_params["head_num"],
            params = params
        )
        
        print("actor: \n", self.actor.policy)
        
        # update
        self.loss = nn.MSELoss(reduce=False)
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, idx, log=False):
        self.actor.policy.train()
        
        self.optimizer.zero_grad()
        loss= self.actor.train_forward(ob_no, ac_na, idx, self.loss, log)

        loss.backward()
        
        # for x in self.actor.policy.parameters():
        #     print(x.grad)
        
        self.optimizer.step()
        
        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        return log
    
    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_index_rollouts(paths)

    def sample(self, batch_size, complete=False):
        '''
            in order for EM algorithms to converge, it must be fitted with the complete dataset
        '''
        return self.replay_buffer.sample_random_data_index(batch_size, complete=complete) 
    
    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_random_data_index(batch_size)

    def save(self, path):
        return self.actor.save(path)  


class VAEMultiHeadAgent(BaseAgent):
    def __init__(self, env, agent_params, params):
        super(VAEMultiHeadAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = VAEMultiHeadPolicy(
            input_shape = self.agent_params['ob_dim'],
            output_shape = self.agent_params['ac_dim'],
            head_num = self.agent_params["head_num"],
            hidden_shape = self.agent_params['size'],
            params = params,
            soft = self.agent_params["soft"]
        )
        
        print("actor: \n", self.actor.policy)
        
        # update
        if self.agent_params["soft"]:
            self.loss = nn.MSELoss(reduce=False)
        else:
            self.loss = nn.MSELoss()
            
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])


    def train(self, ob_no, ac_na, idx, log=False):
        self.actor.policy.train()
        
        self.optimizer.zero_grad()
        loss= self.actor.train_forward(ob_no, ac_na, idx, self.loss, log)

        loss.backward()
        
        # for x in self.actor.policy.parameters():
        #     print(x.grad)
        
        self.optimizer.step()
        
        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        return log
    
    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_index_rollouts(paths)

    def sample(self, batch_size, complete=False):
        '''
            in order for EM algorithms to converge, it must be fitted with the complete dataset
        '''
        return self.replay_buffer.sample_random_data_index(batch_size, complete=complete) 
    
    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_random_data_index(batch_size)

    def save(self, path):
        return self.actor.save(path)   

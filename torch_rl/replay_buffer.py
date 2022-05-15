from utils.utils import *

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions = (
                convert_listofrollouts(paths))

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            # self.rews = rewards[-self.max_size:]
            # self.next_obs = next_observations[-self.max_size:]
            # self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            # if concat_rew:
            #     self.rews = np.concatenate(
            #         [self.rews, rewards]
            #     )[-self.max_size:]
            # else:
            #     if isinstance(rewards, list):
            #         self.rews += rewards
            #     else:
            #         self.rews.append(rewards)
            #     self.rews = self.rews[-self.max_size:]
            # self.next_obs = np.concatenate([self.next_obs, next_observations])[-self.max_size:]
            # self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]

    def sample_random_data(self, batch_size, complete=False):
        assert (
                self.obs.shape[0]
                == self.acs.shape[0]
                # == self.rews.shape[0]
                # == self.next_obs.shape[0]
                # == self.terminals.shape[0]
        )
        
        # if complete, means we train using batch GD
        # that means compute gradients on all training samples
        if complete:
            batch_size = self.obs.shape[0]
            
        idx = np.random.permutation(self.obs.shape[0])
        batch_idx = idx[:batch_size]
        
        obs_batch, acs_batch = [], []
        for i in batch_idx:
            obs_batch.append(self.obs[i])
            acs_batch.append(self.acs[i])
            # rews_batch.append(self.rews[i])
            # next_obs_batch.append(self.next_obs[i])
            # terminals_batch.append(self.terminals[i])
    
        assert len(obs_batch) == len(acs_batch) ==  batch_size #== len(rews_batch) == len(next_obs_batch) == len(terminals_batch) ==
        obs = torch.Tensor(np.array(obs_batch))
        acs = torch.Tensor(np.array(acs_batch))
        # rews = torch.Tensor(rews_batch)
        # next_obs = torch.Tensor(next_obs_batch)
        # terminals = torch.Tensor(terminals_batch)
        # return (obs, acs, rews, next_obs, terminals)
        
        return (obs, acs)
        
    
    
    def add_embedding_rollouts(self, paths, concat_rew=True):
        '''
            add rollouts consisting of (obs, acs, next_obs, terminals, embedding)
        '''
        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals, embedding_input = (
            convert_listofrollouts(paths, concat_rew, embedding_flag=True))

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.embedding_input = embedding_input[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            
            self.next_obs = np.concatenate([self.next_obs, next_observations])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]
            self.embedding_input = np.concatenate([self.embedding_input, embedding_input])[-self.max_size:]
    
    
    def add_index_rollouts(self, paths):
        '''
            add rollouts consisting of (obs, acs, rew, next_obs, terminals, index)
        '''
        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, index_input = (
            convert_listofrollouts(paths, index_flag=True))
        
        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.index_input = index_input[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.index_input = np.concatenate([self.index_input, index_input])[-self.max_size:]
            
        print(self.index_input)
            
            
    def sample_random_data_embedding(self, batch_size):
        '''
            from replay buffer, sample data of batch_size with embedding
        '''
        assert (
                self.obs.shape[0]
                == self.acs.shape[0]
                == self.embedding_input.shape[0]
        )

        idx = np.random.permutation(self.obs.shape[0])
        batch_idx = idx[:batch_size]
        
        obs_batch, acs_batch, embedding_input_batch = [], [], []
        for i in batch_idx:
            obs_batch.append(self.obs[i])
            acs_batch.append(self.acs[i])
            embedding_input_batch.append(self.embedding_input[i])
    
        assert len(obs_batch) == len(acs_batch) == len(embedding_input_batch) == batch_size
        obs = torch.Tensor(obs_batch)
        acs = torch.Tensor(acs_batch)
        embedding_input = torch.stack(embedding_input_batch)
        
        return (obs, acs, embedding_input)


    def sample_random_data_index(self, batch_size, complete=False):
        '''
            from replay buffer, sample data of batch_size with index
        '''
        # print(self.obs.shape[0], self.acs.shape[0], self.index_input.shape[0])
        assert (
                self.obs.shape[0]
                == self.acs.shape[0]
                == self.index_input.shape[0]
        )
        
        # if complete, means we train using batch GD
        # that means compute gradients on all training samples
        if complete:
            batch_size = self.obs.shape[0]

        idx = np.random.permutation(self.obs.shape[0])
        batch_idx = idx[:batch_size]
        
        obs_batch, acs_batch, index_input_batch = [], [], []
        for i in batch_idx:
            obs_batch.append(self.obs[i])
            acs_batch.append(self.acs[i])
            index_input_batch.append(self.index_input[i])
    
        assert len(obs_batch) == len(acs_batch) == len(index_input_batch) == batch_size
        obs = torch.Tensor(obs_batch)
        acs = torch.Tensor(acs_batch)
        # print(index_input_batch)
        index_input = torch.stack(index_input_batch)
        
        return (obs, acs, index_input)


    def sample_recent_data(self, batch_size=1):
        '''
            Only supports single-task environment training
        '''
        return (
            self.obs[-batch_size:],
            self.acs[-batch_size:],
            self.rews[-batch_size:],
            self.next_obs[-batch_size:],
            self.terminals[-batch_size:],
        )


class EnvInfo():
    def __init__(self, 
            env,
            device,
            train_render,
            eval_render,
            epoch_frames,
            eval_episodes,
            max_episode_frames,
            continuous,
            env_rank):

        self.current_step = 0

        self.env = env
        self.device = device
        self.train_render = train_render
        self.eval_render = eval_render
        self.epoch_frames = epoch_frames
        self.eval_episodes = eval_episodes
        self.max_episode_frames = max_episode_frames
        self.continuous = continuous
        self.env_rank = env_rank

        # For Parallel Async
        self.env_cls = None
        self.env_args = None

    def start_episode(self):
        self.current_step = 0

    def finish_episode(self):
        pass
import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 128        
GAMMA = 0.99            
TAU = 1e-3              
LR_ACTOR = 9e-4        
LR_CRITIC = 9e-4       
WEIGHT_DECAY = 0        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, s_update = 20, num_agents = 20, seed = 0):
        
        self.state_size = state_size
        self.action_size = action_size
        self.nagents = num_agents
        
        #Actor networks
        self.actor_local_network = Actor(self.state_size, self.action_size).to(device)
        self.actor_target_network = Actor(self.state_size, self.action_size).to(device)
        
        #Critic networks
        self.critic_local_network = Critic(self.state_size, self.action_size).to(device)
        self.critic_target_network = Critic(self.state_size, self.action_size).to(device)
        
        #optimizers
        self.actor_optimizer = optim.Adam(self.actor_local_network.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_local_network.parameters(), lr=LR_CRITIC, weight_decay = WEIGHT_DECAY)
        
        #Create Noise
        self.noise = Gausian((self.nagents, self.action_size), seed)
        
        #create replay buffer
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.softish_update_every = s_update
        
    def step(self, states, actions, rewards, next_states, dones, t):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.replay_buffer.add(state, action, reward, next_state, done)
        
        if len(self.replay_buffer.memory) > BATCH_SIZE:
#             self.learn(self.replay_buffer.sample(), GAMMA)
            if t > self.softish_update_every:
                self.learn(self.replay_buffer.sample(), GAMMA)
                self.softish_update( self.actor_local_network, self.actor_target_network, TAU)
                self.softish_update( self.critic_local_network, self.critic_target_network, TAU)
    
    
    def act(self, state, add_noise = True):
        noise = self.noise.sample()
        self.actor_local_network.eval()
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = self.actor_local_network(state).to('cpu').data.numpy()
        self.actor_local_network.train()
        return np.clip(action+noise, -1, 1)
    
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        
        states, actions, rewards, next_states, dones = experiences
        
        #critic update
        next_actions = self.actor_target_network(next_states)
        value_next_actions = self.critic_target_network(next_states, next_actions)
        target_value = rewards + gamma*(value_next_actions)*(1-dones)
        expected_value = self.critic_local_network(states, actions)
        
        c_loss = F.mse_loss(expected_value, target_value)
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local_network.parameters(), 1)
        self.critic_optimizer.step()
        
        #actor update (maximize reward)
        actors_guess = self.actor_local_network(states)
        value_estimate = self.critic_local_network(states, actors_guess).mean()
        a_loss = -value_estimate
        
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        
        self.softish_update( self.actor_local_network, self.actor_target_network, TAU)
        self.softish_update( self.critic_local_network, self.critic_target_network, TAU)
    
    def softish_update(self, local, target, tau):
        
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(tau*l_param.data+(1.0-tau)*t_param.data)
        
        
        
class Gausian:
    def __init__(self, size, seed, u=0., t=0.15, s=0.2):
        self.u = u * np.ones(size)
        self.t = t
        self.s = s
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.u)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.t* (self.u - x) + self.s * np.random.standard_normal(self.u.shape)
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

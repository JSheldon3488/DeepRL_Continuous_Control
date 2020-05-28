from networks import Actor, Critic
from utils import OUNoise, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

class DDPG():
    """ Deep Deterministic Policy Gradients Agent used to interaction with and learn from an environment """

    def __init__(self, state_size: int, action_size: int, num_agents: int, random_seed: int):
        """ Initialize a DDPG Agent Object

        :param state_size: dimension of state (input)
        :param action_size: dimension of action (output)
        :param random_seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.t_step = 0
        self.update_every = 10
        self.num_updates = 10

        # Hyperparameters
        self.buffer_size = 1000000
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.001
        self.lr_actor = 0.0001
        self.lr_critic = 0.001
        self.weight_decay = 0

        # Networks
        self.actor_local = Actor(self.state_size, self.action_size,  random_seed).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = self.lr_actor)
        self.critic_local = Critic(self.state_size, self.action_size, random_seed).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size, random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = self.lr_critic, weight_decay = self.weight_decay)
        # Initialize actor and critic networks to start with same parameters
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        # Noise / Exploration Setup
        self.noise = OUNoise(self.action_size, random_seed)

        # Replay Buffer Setup
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def __str__(self):
        return "DDPG_Agent"

    def step(self, states, actions, rewards, next_states, dones):
        """ Save experience in replay memory, and use random sample from buffer to learn. """
        for agent_num in range(self.num_agents):
            self.memory.add(states[agent_num], actions[agent_num], rewards[agent_num], next_states[agent_num], dones[agent_num])

        # Learn num_updates times every "update_every" time step
        self.t_step += 1
        if len(self.memory) > self.batch_size and self.t_step%self.update_every == 0:
            self.t_step = 0
            for _ in range(self.num_updates):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, states, epsilon, add_noise=True):
        """ Returns actions for given state as per current policy """
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval() # Sets to eval mode (no gradients)
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train() # Sets to train mode (gradients back on)
        if add_noise and epsilon > np.random.random():
            actions += [self.noise.sample() for _ in range(self.num_agents)]
        return np.clip(actions, -1,1)

    def reset(self):
        """ resets to noise parameters """
        self.noise.reset()

    def learn(self, experiences):
        """ Update actor and critic networks using a given batch of experiences

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value"""

        states, actions, rewards, next_states, dones = experiences
        # -------------------- Update Critic -------------------- #
        # Use target networks for getting next actions and q values and calculate q_targets
        next_actions = self.actor_target(next_states)
        next_q_targets = self.critic_target(next_states, next_actions)
        q_targets = rewards + (self.gamma*next_q_targets*(1-dones))
        # Compute critic loss (Same as DQN Loss)
        q_expected = self.critic_local(states,actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------- Update Actor --------------------- #
        # Computer actor loss (maximize mean of Q(states,actions))
        action_preds = self.actor_local(states)
        # Optimizer minimizes and we want to maximize so multiply by -1
        actor_loss = -1*self.critic_local(states, action_preds).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #---------------- Update Target Networks ---------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)


    def soft_update(self, local_network, target_network, tau):
        """ soft update newtwork parametes
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_network: PyTorch Network that is always up to date
        :param target_network: PyTorch Network that is not up to date
        :param tau: update (interpolation) parameter
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



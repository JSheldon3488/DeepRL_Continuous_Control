""" This file contains all the utility classes """
import random
import torch
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ReplayBuffer:
    """ Fixed Sized Replay buffer for storing and sampling experience tuples """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Initialize Replay Buffer

        :param action_size:
        :param buffer_size: max size of replay buffer
        :param batch_size: size of training sample
        :param seed: random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.random_seed = random.seed(seed)
        self.memory = deque(max_len = self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """ Add an experience to memory """
        self.memory.append(self.experience(state,action,reward,next_state,done))

    def sample(self):
        """ Sample batch_size of experiences randomly from memory """
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        " Return the current size of internal memory "
        return len(self.memory)
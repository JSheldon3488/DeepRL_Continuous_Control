import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """ Takes in a network layer and returns a range to use for parameter initialization """
    fan_in = layer.eeight.data.size()[0]
    lim = 1./ np.sqrt(fan_in)
    return (-lim,lim)

class Actor(nn.Module):
    " Actor (Policy) network for action selection "
    def __init__(self, state_size, action_size, seed):
        """ Initialize parameters and build actor network

        :param state_size: Dimension of the input state
        :param action_size: Dimension of the output state
        :param seed: random seed
        """
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400,300)
        self.fc3 = nn.Linear(300, action_size)
        self.reset_parameters()

    def forward(self, state):
        """ policy network that maps states -> actions """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

    def reset_parameters(self):
        """ Initialize or reset parameters for the network """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)


class Critic(nn.Module):
    " Critic (Value) network for evaluating actions "

    def __init__(self, state_size, action_size, seed):
        """ Initialize the parameters and set up the network

        :param state_size: Dimension of input state
        :param action_size: Dimension of output state
        :param seed: random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400+action_size,300)
        self.fc3 = nn.Linear(300, 1)
        self.reset_parameters()

    def forward(self, state, action):
        """ Critic Network that maps (state,action) pairs -> Q-Values """
        temp = F.relu(self.fc1(state))
        x = torch.cat((temp,action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def reset_parameters(self):
        """ Initialize or reset parameters for the network """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)
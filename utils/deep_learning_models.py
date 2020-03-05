import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __capacity__(self):
        return self.capacity


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(49, 16)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x1, x2):
        y1 = F.relu(self.fc1(x1.float()))
        y1 = F.relu(self.fc3(y1))
        y2 = F.relu(self.fc2(x2.float()))
        if y1.size()[0] > 10:
            z = torch.cat((y1, y2), dim=1)
        else:
            z = torch.cat((y1, y2))
        out = F.softmax(self.fc4(z))
        return out

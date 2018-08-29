import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.utils.data
import torchvision
from torch.autograd import Variable
from torch import optim

import numpy as np
import os
import time
import datetime
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


class LocationEncoder(nn.Module):
    def __init__(self, pedestrian_num, input_size, hidden_size):

        self.layer_num = pedestrian_num
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, hidden_size)

    def forward(self, input):
        # input = [pedestrain_num, input_size]
        hidden = []

        for idx in range(0, self.layer_num):
            input_data = input[idx]
            hidden_data = F.relu(self.fc1(input_data))
            hidden_data = F.relu(self.fc2(hidden_data))
            hidden_data = self.fc3(hidden_data)

            hidden.append(hidden_data)
        return hidden

class MontionEncoder(nn.Module):
    def __int__(self):
        print('aa')
    def forward(self, *input):
        print('hi')

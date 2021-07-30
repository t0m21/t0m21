#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:40:22 2020

@author: tes520
"""


import torch.nn as nn
import torch.nn.functional as F



class NN(nn.Module):
    
    def __init__(self, input_size, num_classes):
        
        #initialize super
        super().__init__()
        
        #define layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, num_classes)


    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return F.softmax(x, dim = 1)
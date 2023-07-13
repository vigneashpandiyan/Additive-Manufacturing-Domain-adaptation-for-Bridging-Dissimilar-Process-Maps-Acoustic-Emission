# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv

"""

#%% Libraries to import

import torch
import torch.nn.init
import torch.nn as nn
from torch.nn import functional as F

# a small value
EPSILON = 1e-8
'''
Network proposed in this work
'''
class Network(nn.Module):


    def __init__(self,EMBEDDING_DIM,dropout_rate):
        super(Network, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=16) 
        self.bn1 = nn.BatchNorm1d(4)
        
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=16) 
        self.bn2 = nn.BatchNorm1d(8) 
        
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16) 
        self.bn3 = nn.BatchNorm1d(16) 
        
        self.conv4 = nn.Conv1d(16, 32, kernel_size=16)
        self.bn4 = nn.BatchNorm1d(32) 
        
        self.conv5 = nn.Conv1d(32, 64, kernel_size=16)
        self.bn5 = nn.BatchNorm1d(64) 
                       
        # self.fc1 = nn.Linear(832, EMBEDDING_DIM)
       
        self.pool = nn.MaxPool1d(3)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        
        x = F.relu(self.bn5(self.conv5(x)))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        # x = x.view(-1, 832)
        x = x.view(x.shape[0], -1)
       
        
        return x
    


class Associative_Regularizer_loss(nn.Module):

    def __init__(self):
        super(Associative_Regularizer_loss, self).__init__()

    def forward(self, a, b, labels_for_a):
        """
        Arguments:
            a: a float tensor with shape [n, d].
            b: a float tensor with shape [m, d].
            labels_for_a: a long tensor with shape [n],
                it has values in {0, 1, ..., num_labels - 1}.
        Returns:
            two float tensors with shape [].
        """
        d = a.size(1)
        d_a = a.size()
        d_bt =  b.t().size()
       
        p = torch.matmul(a, b.t())  # shape [n, m]
        dp = p.size()
        
        p /= torch.tensor(d).float().sqrt()
        dpa = p.size()
        
        ab = F.softmax(p, dim=1)  # shape [n, m]
        dab = ab.size()
       
        ba = F.softmax(p.t(), dim=1)  # shape [m, n]
        dba = ba.size()
        
        
        aba = torch.matmul(ab, ba)  # shape [n, n]
        daba = aba.size()
       
        
        labels = labels_for_a.unsqueeze(0)  # shape [1, n]
        
        is_same_label = (labels == labels.t()).float()  # shape [n, n]
        
        label_count = is_same_label.sum(1).unsqueeze(1)  # shape [n, 1]
        
        targets = is_same_label/label_count  # shape [n, n]
        
        Associative_loss = targets * torch.log(EPSILON + aba)  # shape [n, n]
        Associative_loss = Associative_loss.sum(1).mean(0).neg()

        visit_probability = ab.mean(0)  # shape [m]
        # note that visit_probability.sum() = 1

        m = b.size(0)
        targets = (1.0 / m) * torch.ones_like(visit_probability)
        Regularizer_loss = targets * torch.log(EPSILON + visit_probability)  # shape [m]
        Regularizer_loss = Regularizer_loss.sum(0).neg()

        return Associative_loss, Regularizer_loss


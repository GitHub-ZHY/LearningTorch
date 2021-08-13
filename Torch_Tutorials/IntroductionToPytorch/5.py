# _*_ coding: utf-8 _*_
# @File : 5.py
# @Desc : 
# @Time : 2021/8/4 10:47 
# @Author : HanYun.
# @Version：V 1.0
# @Software: PyCharm
# @Related Links:

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
print(X.shape)
loggits = model(X)
pred_probab = nn.Softmax(dim=1)(loggits)
print(pred_probab.sum(0))
print(pred_probab.sum(1))   # 解释了上句 dim=1 的含义
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")



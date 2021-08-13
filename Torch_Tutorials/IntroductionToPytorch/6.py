# _*_ coding: utf-8 _*_
# @File : 6.py
# @Desc : https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
# @Time : 2021/8/4 12:49 
# @Author : HanYun.
# @Version：V 1.0
# @Software: PyCharm
# @Related Links:

import torch
import torch.nn.functional
import torch.nn.functional as F

x = torch.ones(5)  # input tensor
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = F.binary_cross_entropy_with_logits(z, y)
# loss2 = torch.nn.BCELoss(z, y) # 同上

print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

loss.backward()
print(w.grad, '\n', b.grad)

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

inp = torch.eye(5, requires_grad=True)
out = (inp + 1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)

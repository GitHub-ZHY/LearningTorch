# _*_ coding: utf-8 _*_
# @File : 8.py
# @Desc : 
# @Time : 2021/8/4 15:40 
# @Author : HanYun.
# @Version：V 1.0
# @Software: PyCharm
# @Related Links:

import torch
import torch.onnx as onnx
import torchvision.models as models

# 保存权重
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
# 加载权重，先创建同类型的实例
model = models.vgg16()
model.load_state_dict(torch.load('model_weight.pth'))
model.eval()

# Saving and Loading Models with Shapes
torch.save(model,'model.pth')

model = torch.load('model.pth')



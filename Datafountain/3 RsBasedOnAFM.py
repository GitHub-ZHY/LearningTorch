# _*_ coding: utf-8 _*_
# @File : 3 RsBasedOnAFM.py
# @Desc : 
# @Time : 2021/9/17 14:41 
# @Author : HanYun.
# @Version：V 1.0
# @Software: PyCharm
# @Related Links:

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# DEBUG = True
# 参数设置，无GPU将参数改为False
class Config(object):
    def __init__(self):
        self.path = './data/3 ml-latest-small(datafountain)'
        self.batch_size = 1024
        self.GPU_available = False
        self.embedding_dim = 70
        self.epoch = 10
        self.lr = 5e-3


class Data(object):
    # 案例以数据集中的tags.csv来构建数据集，
    # 以其中的user、movie、tag为特征，进行label-encoder，构建出数值特征，
    # 然后对数据进行负采样，每个正例负采样两个数据，构建出完整数据集，
    # 然后以8:2的比例划分成训练集(ml-tag.train.txt)、测试集(ml-tag.test.txt)。
    # 对于有tag记录的数据label为1，负采样的为-1。
    def __init__(self, config, process_type):
        self.process_type = process_type
        self.path = config.path + '/ml-tag.' + process_type + '.txt'
        self.batch_size = config.batch_size
        self.ToTensor = torch.LongTensor
        self.num_features = 0
        if config.GPU_available:
            self.ToTensor = torch.cuda.LongTensor
        self.group, self.label, self.num_features = self.readFile(self.path)

    # Read data from local files
    def readFile(self, path):
        groups = []
        labels = []
        num_features = 0
        with open(path, 'r') as f:
            while True:
                line = f.readline().strip('\n').split(" ")
                if line == None or line == ['']:
                    break
                group = [float(i) for i in line[1:]]  # [:-2],即去掉采样标注的信息，如 '1' '-1'
                num_features = max(num_features, max(group))
                groups.append(group)
                labels.append(float(line[0]))
        return groups, labels, int(num_features)

    # 构建 torch 数据集，形成 dataloader, 用于模型训练
    def dataloader(self, process_type):
        data = TensorDataset(self.ToTensor(self.group))
        if process_type == 'train':
            return DataLoader(data, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(data, batch_size=len(self.group), shuffle=False)

    def feature_number(self):
        return self.num_features


class Attention(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(Attention, self).__init__()
        # 全连接层
        self.L1 = nn.Linear(embedding_dim, 30)
        self.L2 = nn.Linear(30, 1)

    # 定义模型数据流
    def forward(self, x):
        out = self.L1(x)
        out = nn.Sigmoid()(out)  # 先创建对象，自动构造，随后调用。相当于 s = nn.Sigmoid(); out = s(out)
        out = self.L2(out)
        out = nn.Sigmoid()(out)
        return out


class afm(nn.Module):
    def __init__(self, config, num_features):
        super(afm, self).__init__()
        self.ToTensor = torch.LongTensor
        self.GPU = config.GPU_available
        if self.GPU:
            self.ToTensor = torch.cuda.LongTensor
            self.device = torch.device('cuda')

        self.person_embed = nn.Embedding(num_features + 1, config.embedding_dim)  # embedding层定义
        self.prediction = nn.Linear(config.embedding_dim, 1)  # 全连接层
        self.attention = Attention(config.embedding_dim)  # attention层实现

        self.p = nn.Parameter(torch.rand(1, config.embedding_dim))
        self.first = nn.Embedding(num_features + 1, 1)
        self.gen_bias = nn.Parameter(torch.rand(1))

    # 模型的数据流
    def forward(self, group, config):
        first = self.FirstOrder(group, config)
        second = self.AttentiveSecond(group, config)
        asd = nn.Sigmoid()(first + second) * 2 - 1
        return asd

    # 数据流中函数详细定义
    def AttentiveSecond(self, group, config):
        gr_len = len(group)
        gr_embed = []
        a = torch.FloatTensor()
        if config.GPU_available:
            a = torch.cuda.FloatTensor()

        for x in group:
            gr_embed.append(self.person_embed(self.ToTensor(x)))

        for gr in gr_embed:
            b = torch.FloatTensor()
            if config.GPU_available:
                b = torch.cuda.FloatTensor()

            v_ij = torch.Tensor()
            a_ij = torch.Tensor()

            if config.GPU_available:
                v_ij = v_ij.to(self.device)
                a_ij = a_ij.to(self.device)

            for i, v_i in enumerate(gr):
                for v_j in gr[i + 1:]:
                    element_wise = torch.mul(v_i, v_j)
                    v_ij = torch.cat((v_ij, element_wise.reshape(-1, 1)), dim=-1)
                    a_ij = torch.cat((a_ij, self.attention(element_wise)))

            v_ij = v_ij.reshape(config.embedding_dim, -1)
            a_hat_ij = nn.Softmax(dim=0)(a_ij)  # relative impact

            product = torch.mm(self.p, torch.mul(v_ij, a_hat_ij)).sum()
            a = torch.cat((a, product.reshape(1)))
        #        a = nn.Softmax(dim=0)(a.flatten())
        return a


config = Config()
trainData = Data(config, 'train')
testData = Data(config, 'test')

device = torch.device('cuda')
# num_features = trainData.feature_number()
# print(trainData.feature_number())
# print(num_featrues)
model = afm(config, trainData.feature_number())
if config.GPU_available:
    model = model.to(device)

lr = config.lr
optimizer = optim.Adam(model.parameters(), lr)
loss_fc = nn.MSELoss()  # MSE损失函数


# 模型训练函数的定义
def train(model, data, config):
    total_loss = 0
    loss = 0
    for batch_id, (group, label) in enumerate(data):
        if config.GPU_available:
            group = group.to(device)
            label = label.to(device)

        model.zero_grad()
        output = model(group, config)
        loss = loss_fc(output, label)
        total_loss += loss
        loss.backward()
        optimizer.step()
        # print('batch_id : %d, loss : %f' %(batch_id,loss))
    total_loss /= (batch_id + 1)
    print('epoch loss:\t{}'.format(total_loss.item()))
    return (total_loss.item())


# 模型评测函数的定义
def evaluate(model, data, config):
    with torch.no_grad():
        loss = 0
        for _, (group, label) in enumerate(data):
            if config.GPU_available:
                group = group.to(device)
                label = label.to(device)
            output = model(group, config)
            loss = loss_fc(output, label)
        print("MSELoss:\t{}".format(loss))
        return loss.item()


Loss_train = []
Loss_test = []
minMSE = 99999999
Time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# 将训练中的输出写入到日志文件
with open('./temp/3-{}.log'.format(Time), 'w') as f:
    for epoch in range(config.epoch):
        t1 = time.time()
        model.train()
        print('epoch:\t{}'.format(epoch + 1))
        f.write('epoch:\t{}\n'.format(epoch + 1))
        MSE = train(model, trainData.dataloader('train'), config)
        t2 = time.time()
        f.write('epoch loss:\t{},\ttrain_time:\t{}\n'.format(MSE, t2 - t1))
        Loss_train.append(MSE)
        model.eval()
        testData = Data(config, 'test')
        print("evaluate:")
        f.write("evaluate:\t")
        t1 = time.time()
        MSE = evaluate(model, testData.dataloader('test'), config)
        if MSE < minMSE:
            torch.save(model.state_dict(), './temp/{}_{:.4f}.model'.format(Time, MSE))
            minMSE = MSE
        Loss_test.append(MSE)
        f.write("MSELoss:\t{},\tevaluate_time:\t{}\n".format(MSE, time.time() - t1))

loadModel = model.load_state_dict(torch.load('./temp/{}_{:.4f}.model'.format(Time, minMSE)))
print(loadModel)

plt.figure()  # 初始化画布
x1 = range(0, config.epoch)  # 取横坐标的值
plt.plot(x1, Loss_train, label='train_MSE')  # 绘制折线图
plt.scatter(x1, Loss_train)  # 绘制散点图
plt.plot(x1, Loss_test, label='test_MSE')
plt.scatter(x1, Loss_test)
plt.xlabel('Epoch #')  # 设置坐标轴名称
plt.ylabel('MSE')
plt.legend()
plt.show()  # 显示图片

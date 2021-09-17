# _*_ coding: utf-8 _*_
# @File : mytransformer.py
# @Desc : 
# @Time : 2021/8/15 21:07 
# @Author : HanYun.
# @Version：V 1.0
# @Software: PyCharm
# @Related Links:

# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")


# % matplotlib inline # 朱皮特用这个  解决：删掉或者这行代码，用 plt.show() 展示图表。


# region 1 模型架构类
#     EncoderDecoder类包含两个架构 Encoer和 Decoder，
#     前向传播包含两个架构，同时也将两个架构单独定义，用于后面model.eval()模型评估。
#     其输入包含encoder,decoder后面详细介绍，
#     src_embed, tgt_embed分别为输入和目标输出的enbedding形式，
#     最后一个参数generator是用于将模型最后训练结果转化为概率值，实际是linear+softmax激活。
class EncoderDecoder(nn.Module):
    """
    A Standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# endregion

# region 2.1 Encoder
# Encoder架构是将多层EncoderLayer连接，这里用到clones函数，复制多个layer返回ModuleList
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# endregion

# region 2.2 EncoderLayer

# 编码器每个模块有两个子层：multi-head自注意力层和逐位置全连接前馈网络。
# 其中，这两个子层都有包含残差连接SublayerConnection，其中残差连接前会进行LayerNorm-层归一化。实际上是对本层的输入进行norm，这里从代码可以看出，模型的输入其实也是做了一个norm。
# 注意：层均一化与BN归一化不同，BN是对一批样本的每个特征分别进行归一化，在(N,C,H,W)的四维张量里，一个特征不是一个像素，而是一个通道。因此要对N,H,W三个维度进行归一化。LN针对的是每一个batch的每一层的神经元的输入，因此不依赖于batch和sequence长度，一个特征是一个单词，代码中体现也就是最后一个维度。
# LN是针对一个样本来做的均值归一化，BN针对一个batch
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# endregion

# region 2.3 Decoder
# Decder与 Encoder类似，不同点在于 DecoderLayer一层中有两个自注意力子层，
# 第二个自注意力子层 key,value连接 Encoder的输出，
# 即编码层的的输出从代码来看K和V相同，与编码层类似，在多头自注意力模型中会乘一个变换矩阵。
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# 前面提到，输入包含mask，修正编码器层中的自注意力子层，以防止当前位置注意到后续序列位置。
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  #
    return torch.from_numpy(subsequent_mask) == 0  # 返回上三角为0的torch Tensor


# test查看
plt.figure(figsize=(5, 5))
plt.imshow(subsequent_mask(20)[0])
plt.show()


# endregion

# region 2.4 Multi-Head Attention
# Multi-Head Attention分为两点，多头和自注意力。
# 自注意力可以通过下面函数实现，其实际上就是两个矩阵点乘，输入包含query,key,value,mask。
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # (batch_size, heads, max_seq_len, d_k) * (batch_size, heads, d_k, max_seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # 对于padding部分，赋予一个极大的负数，softmax后该项的分数就接近0了，表示贡献很小
        # masked_fill(mask,value)在mask为1的地方填充
        scores = scores.masked_fill(mask == 0, -1e9)  # masked_fill(mask)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # (batch_size, heads, max_seq_len, max_seq_len) * (batch_size, heads, max_seq_len, d_k)
    # = (batch_size, heads, max_seq_len, d_k)
    return torch.matmul(p_attn, value), p_attn


# 有了自注意力计算函数，就可以得到下面的自注意力类
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # batch_size

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # view转换数据的维度
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # 返回一个内存连续的有相同数据的tensor
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# endregion

# region 2.5 Utils
# 逐位置的前馈网络
# 两个全连接层
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# 词嵌入
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 位置编码
# 位置编码是Transformer模型中最后一个需要注意的结构，它对使用注意力机制实现序列任务也是非常重要的部分。
# 如上文所述，Transformer使用自注意力机制抽取序列的内部特征，但这种代替RNN或CNN抽取特征的方法有很大的局限性，即它不能捕捉序列的顺序。
# 这样的模型即使能根据语境翻译出每一个词的意义，也组不成完整的语句。

# 为了令模型能利用序列的顺序信息，我们必须植入一些关于词汇在序列中相对或绝对位置的信息。
# 直观来说，如果语句中每一个词都有特定的位置，那么每一个词都可以使用向量编码位置信息。
# 将这样的位置向量与词嵌入向量相结合，那么我们就为每一个词引入了一定的位置信息，注意力机制也就能分辨出不同位置的词。
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # [max_len,d_model]
        position = torch.arange(0., max_len).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))  # [1,d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1,max_len,d_model]
        self.register_buffer('pe', pe)  # 注册buffer,不会更新参数

    def forward(self, x):  # x = [1,wordnum,d_model]
        # 位置编码 + 词向量 x.size(1)为单词的个数
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# endregion

# region 2.6 构建模型
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


# 下面我们定义一个批处理对象，它包含用于训练的src和目标句，以及构造掩码
class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src  # （batch，单词数）
        self.src_mask = (src != pad).unsqueeze(-2)  #
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]  # target 评估
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        # 隐藏 pad + future words
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)  # 不为0（pad）记为1 ==> mask
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))  # 1并1才为1，所以有一个为0则为0
        return tgt_mask


# endregion

# region 2.7 训练和构建 batch
# 以下定义的一个计算batch_size的函数，在测试阶段暂时用不到。
global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


# optimizer
# 计算step，根据step更新学习率
# Note: This part is incredibly important.
# Need to train with this setup of the model is very unstable.
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# test查看优化模型在不同模型尺寸和优化超参数的学习率
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
plt.show()


# None

# LabelSmoothing
# LabelSmoothing实际是防止预测结果过于自信，添加了一个通常为均匀分布的噪声。
# 代码如下 https://blog.csdn.net/weixin_40548136/article/details/100582631
# https://blog.csdn.net/lqfarmer/article/details/74276680
class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size

        true_dist = x.data.clone()  # 复制x

        true_dist.fill_(self.smoothing / (self.size - 2))  # e*u(k)填充

        true_dist.scatter_(1, target.data.unsqueeze(1),
                           self.confidence)  # scatter_(input, dim, index, src) 按行dim=1赋值，index以target为准 赋的值(1-smooth)*q(y|x)

        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)  # [[2]]获取target数据等于padding_的索引

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)  # index_fill_(dim,index,val)在dim维度填充index为2值为0

        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))  # KL 散度


# 以下可以查看LabelSmoothing对预测值损失的计算影响，这里可以打印实际输出的真值，查看噪声影响。
# Example of label smoothing. 可视化真值分布
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()),
         Variable(torch.LongTensor([2, 1, 0])))
# Show the target distributions expected by the system. 真值分布
plt.imshow(crit.true_dist)
plt.show()
# None

# 标签平滑实际上在模型对某些选项非常有信心的时候会惩罚它。
# x增大在一定程度上 loss增大
crit = LabelSmoothing(5, 0, 0.2)


def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ])
    # print(predict)
    return crit(Variable(predict.log()),
                Variable(torch.LongTensor([1]))).item()


plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
plt.show()


# endregion

# region 2.8 样例数据
# 合成数据
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))  # （batch=batch_size，10）
        data[:, 0] = 1  # 这里感觉不是填充，而是start_symbol
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


# 计算损失
# 损失计算
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm


# 模型训练
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

model = make_model(V, V, N=2)
# 优化器
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()  # 训练模式
    # run_epoch(data_gen(V,batch,nbatch),model,losscompute)
    run_epoch(data_gen(V, 30, 20), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))


# 模型评估
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


model.eval()  # 评估模式
src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
src_mask = Variable(torch.ones(1, 1, 10))
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
# endregion

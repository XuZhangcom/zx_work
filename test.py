import numpy as np
import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

import random

import tools

def test1():
    #绘图
    def f(x):
        return 3 * x ** 2 - 4 * x
    x = np.arange(0, 3, 0.1)
    d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
    d2l.plt.savefig('savefig_example.png')

    #计时
    n = 10000
    a = torch.ones(n)
    b = torch.ones(n)

    c = torch.zeros(n)
    timer = tools.Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'{timer.stop():.5f} sec')

    timer.start()
    d = a + b
    print(f'{timer.stop():.5f} sec')

def synthetic_data(w, b, num_examples):#@save
    #生成y=Xw+b噪声
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train=True): #@save
    #构造一个Pytorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2 
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features', features[0], '\nlabel:', labels[0])

    d2l.set_figsize()
    d2l.plt.scatter(features[:,(1)].detach().numpy(), labels.detach().numpy(), 1)
    d2l.plt.savefig('input.png')

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))

    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    print('w的估计误差: ', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差: ', true_b - b)


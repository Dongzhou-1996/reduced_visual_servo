import torch
# 开发时间：2020/9/22 15:28
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

#建立模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(1,1)   #输入输出维数(in_channels,out_channels,kernel_size)
        self.linear2 = nn.Linear(1,1)
    def forward(self,x):
        out = self.linear(x)
        out2 =self.linear2(out)
        return out2

#测试数据
x_train = np.array([[3.4],[5.4],[6.71],[6.89],[6.93],[4.16],[9.779],[6.182],[7.59]
                   ,[2.167],[7.042],[10.791],[5.31],[7.997],[3.1]],dtype = np.float32)

y_train = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],
                   [1.221],[2.87],[3.45],[1.6],[2.9],[1.3]],dtype = np.float32)
#我们想要做的是找一条直线去逼近这些点，希望这条直线离这些点的距离之和最小
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
plt.plot(x_train,y_train,'ro')
plt.show()

#这里有个问题，就是模型和数据，要都分别放入GPU吗
# if torch.cuda.is_available():
# #     model = LinearRegression().cuda()#把这个模型换个名字
# #     inputs = Variable(x_train).cuda()#定义输入
# #     target = Variable(y_train).cuda()#定义目标
if torch.cuda.is_available():
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    model = LinearRegression().cuda()#把这个模型换个名字
    inputs = x_train.cuda()#定义输入
    target = y_train.cuda()#定义目标
else:
    model = LinearRegression()
    inputs = Variable(x_train)
    target = Variable(y_train)

criterion = nn.MSELoss()            #损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 100000
for epoch in range(num_epochs):
    #forward
    out = model(inputs)             #得到前向传播的结果
    loss = criterion(out,target)    #求得输出和真实目标之间的损失函数

    #backward
    optimizer.zero_grad()           #归零梯度，每次反向传播前都要归零梯度，不然梯度会累积，造成结果不收敛
    loss.backward()                 #反向传播
    optimizer.step()                #更新参数
    if (epoch + 1) % 200 == 0:
        print('Epoch[{}/{}],loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))
    #这里的loss.item()，没有按照书上写的
    #这句话目的：在训练过程中过一段时间就将损失函数的值打印出
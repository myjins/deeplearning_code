import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# bn1+relu1+[(conv1+bn2+relu2+droprate+conv2)+(conv1)]=return
class BasicBlock(nn.Module):
    # 初始话，参数传入
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        # 批归一化
        self.bn1 = nn.BatchNorm2d(in_planes)
        # relu即刻激活
        self.relu1 = nn.ReLU(inplace=True)
        #卷积1，3x3，填充1，步长,输入输出通道数为传入参数，无偏置
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # bn+relu+conv2
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # droprate
        self.droprate = dropRate
        # 输入输出通道数是否一致，不一致通过1x1卷积为之前输出通道值
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes,
                                                    kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    # 前向传播函数，张量残差处理，主流程
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        # senet ok
        # sk ok
        # cbam
        # danet
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
# 批量创建残差块，残差块数量nb_layers，残差块类型block
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, dropRate)
    # 内部函数，创建指定类型残差块并串联
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    dropRate):
        # 定义残差块空列表
        layers = []
        # 循环参数输入
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropRate))
        # *列表/元组解包运算符，nn.Sequential构建串联神经网络
        return nn.Sequential(*layers)
    # 张量输入，经所有残差块输出
    def forward(self, x):
        return self.layer(x)
#模型设计
class WideResNet(nn.Module):
    # 深度，分类类别数，宽度
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        # 4：初始cov，bn，全连接，全局池化/激活
        # 6：3（block）*2（核心卷积层）
        # n：BasicBlock数量
        # 保证深度参数值满足wideresnet规则，depth=4+6n，不满足直接报错
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        # 定义block类型
        block = BasicBlock
        # 1st conv：将输入3通道转换为初始16
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 3个block，16到16w到32w到64w
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block,
                                   1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block,
                                   2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block,
                                   2, dropRate)
        # global average pooling and classifier对三次block输出进行bn和激活
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        # 输入向量纬度，分类类别数
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        # 参数初始化，self.modules()遍历WideResNet中所有的子层/子模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # he初始化，让权重均值为0方差为1，mode标准差的计算标准，nonlinearity指定适配激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # 保证不改变特征
            elif isinstance(m, nn.BatchNorm2d):
                # 缩放系数为1
                m.weight.data.fill_(1)
                # 偏移系数为0
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                # 偏置为0，默认为0增加可读性
                m.bias.data.zero_()
    def forward(self, x):
        # 卷积，block1，2，3，bn+relu，平均池化，展为二维，全连接
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out,8)
        # -1：自动计算维度
        out = out.view(-1, self.nChannels)
        return self.fc(out)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()

        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, height*width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()

        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()


        proj_query = x.view(m_batchsize, C, -1)
        proj_key   = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)

        # 数值稳定（官方写法）
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DANet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        inter_channels = in_channels // 4

        # 分支1
        self.conv5a = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # 分支2
        self.conv5c = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # 注意力模块
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)

        # 后处理
        self.conv51 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.conv52 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, x):
        # PAM 分支
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        # CAM 分支
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        # 融合
        feat_sum = sa_conv + sc_conv
        output = self.conv6(feat_sum)

        return output

# bn1+relu1+[(conv1+bn2+relu2+droprate+conv2)+(conv1)]=return
class BasicBlock(nn.Module):
    # 初始话，参数传入
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0,reduction=16):
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

        self.bn3=nn.BatchNorm2d(out_planes)
        # 输入输出通道数是否一致，不一致通过1x1卷积为之前输出通道值
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes,
                                                    kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.se=DANet(out_planes,out_planes)
    # 前向传播函数，张量残差处理，主流程
    def forward(self, x):
        # residual = x
        # out=self.relu1(self.bn1(x))
        # out=self.relu2(self.bn2(self.conv1(out)))
        # if self.droprate > 0:
        #     out = F.dropout(out, p=self.droprate, training=self.training)
        # out = self.conv2(out)
        # out=self.bn3(out)
        # out=self.se(out)
        # if self.equalInOut:
        #     out=residual+
        #
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out=self.bn3(out)
        out=self.se(out)
        out=torch.add(x if self.equalInOut else self.convShortcut(x), out)
        return out
# 批量创建残差块，残差块数量nb_layers，残差块类型block
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0,reduction=16):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, dropRate,reduction)
    # 内部函数，创建指定类型残差块并串联
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    dropRate,reduction=16):
        # 定义残差块空列表
        layers = []
        # 循环参数输入
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropRate,reduction))
        # *列表/元组解包运算符，nn.Sequential构建串联神经网络
        return nn.Sequential(*layers)
    # 张量输入，经所有残差块输出
    def forward(self, x):
        return self.layer(x)
#模型设计
class WideResNet(nn.Module):
    # 深度，分类类别数，宽度
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0,reduction=16):
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
                                   1, dropRate,reduction)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block,
                                   2, dropRate,reduction)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block,
                                   2, dropRate,reduction)
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







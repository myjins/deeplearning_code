# 日志和数据集的路径要改
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

# from senet_s import WideResNet
from sknet_s import WideResNet
# used for logging to TensorBoard
from torch.utils.tensorboard import SummaryWriter
#from tensorboard_logger import configure, log_value

# 命令参数配置
parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
# 通过此参数指定数据集，，参数默认值，参数类型，说明
# 数据集
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
# epoch
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
# 批次大小
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
# 初始学习率
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
# 动量，加速收敛，减少震荡
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
# 提高动量性能
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
# 权重衰减，防止过拟合
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
# 每训练多少个批次打印一次日志
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
# 总层数
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
# 扩宽因子
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
# 丢弃率
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
# 数据增强，随机裁剪旋转等
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
# 断点续训路径，若训练中断，设为--resume checkpoint.pth可加载权重继续训练
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
# 实验名称
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
# tensorboard日志，可视化loss/精度曲线，方便分析训练过程
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
# 数据增强默认值
parser.set_defaults(augment=True)

# 最高精度初始化
best_prec1 = 0
# 日志存储路径
Log_dir='G:/pycharm acaconda/WideResNet-pytorch-master/end/senet_srizhi'#这里需要修改
writer = SummaryWriter(Log_dir)

def main():
    global args, best_prec1
    # 参数解析
    args = parser.parse_args()

    # Data loading code
    # 数据归一化定义，红绿蓝均值，CIFAR-10
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    # 基于是否进行数据增强的训练集数据处理
    if args.augment:
        transform_train = transforms.Compose([
            # 转换为chw格式并进行数值缩放
        	transforms.ToTensor(),
            # 填充为3x40x40，pad（左右上下），mode填充方式reflect镜像填充无黑边
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						 (4,4,4,4),mode='reflect').squeeze()),
            # 转为pil，适配随即增强api
            transforms.ToPILImage(),
            # 随机选择32x32的区域
            transforms.RandomCrop(32),
            # 概率为0.5的左右镜像翻转
            transforms.RandomHorizontalFlip(),
            # PIL的H×W×C到PyTorch标准的C×H×W，数值缩放和类型转换
            transforms.ToTensor(),
            # 归一化
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    # 测试集
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    # 数据加载的配置参数，0适配Windows，true将加载数据直接拷贝到GPU可访问的固定内存区域（先存在普通内存）
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # 检查数据集是否正确
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    # 加载数据集
    train_loader = torch.utils.data.DataLoader(
        # 数据保存路径，训练/测试集，自动下载，数据预处理
        datasets.__dict__[args.dataset.upper()]('G:/pycharm acaconda/WideResNet-pytorch-master/end/data', train=True, download=True,
                         transform=transform_train),
        # 批量大小，打乱训练集顺序，参数配置解包
        batch_size=args.batch_size, shuffle=True, **kwargs)
    # 测试集
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('G:/pycharm acaconda/WideResNet-pytorch-master/end/data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model，深度，根据数据集决定最后输出维度，通道数（宽度因子）
    model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                            args.widen_factor, dropRate=args.droprate,reduction=16)

    #返回所有可训练参数并计算单个参数张量元素总和，可根据输出数据进行评估
    print('Number of model parameters: {}'.format(
      sum([p.data.nelement() for p in model.parameters()])))
    # 将训练移到gpu
    # 多gpu
    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    # 单gpu
    model = model.cuda()

    # optionally resume from a checkpoint
    # 断点续训
    if args.resume:
        # 是否有快照
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # 加载，恢复训练轮数，最优精度，模型权重
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # cudnn，Gpu加速库选择最适配硬件的算法
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    # 损失函数，并移至gpu
    criterion = nn.CrossEntropyLoss().cuda()
    # 优化器（优化参数，初始学习率）
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)

    # cosine learning rate余弦退火学习率调度器，优化器，余弦曲线周期（总步数）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        # 训练数据加载器，模型，损失函数，优化器，学习率，训练轮数
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        # 测试数据，模型，损失函数，训练轮数
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""
    # 批次训练耗时
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    # self.training = True，调用 model.eval()则self.training = False
    model.train()

    # 当前时间
    end = time.time()
    # 图片张量，便签张量
    # enumerate 固定返回 (索引, 元素)
    for i, (input, target) in enumerate(train_loader):
        # 将数据转到gpu
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output = model(input)
        # 计算损失
        loss = criterion(output, target)

        # measure accuracy and record loss
        # 计算精度，[0]取top1精度
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        # 用AverageMeter更新损失，平均损失和样本数
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        # 梯度清零，防止下一轮叠加
        optimizer.zero_grad()
        # 反向传播，计算新梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 学习率调整
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # 打印监控日志
        if i % args.print_freq == 0:
            # epoch，epoch内批次i，总epoch，耗时，损失，精度
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    # log to TensorBoard
    # TensorBoard日志
    if args.tensorboard:
        writer.add_scalar('train_loss', losses.avg, epoch)
        writer.add_scalar('train_acc', top1.avg, epoch)

# 测试模块
def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        #禁用梯度计算
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('val_loss', losses.avg, epoch)
        writer.add_scalar('val_loss', losses.avg, epoch)

    return top1.avg

# 训练参数核心，布尔值，文件名
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    # args.name：argparse库，配合-name起名
    directory = "runs/%s/"%(args.name)
    # 目录不存在创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    #拼接路径
    filename = directory + filename
    # 保存到指定路径
    torch.save(state, filename)
    # 如果是最优的话，复制
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    # 当前批次值，均值，加权总和，加权计数，方便初始化置零
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # 本轮更行计算
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # 最大k
    maxk = max(topk)
    batch_size = target.size(0)
    # 返回为maxk的最高值和类别索引，k，1维，降序排序，返回是否为原始位置索引
    _, pred = output.topk(maxk, 1, True, True)
    # 行列转置
    pred = pred.t()
    # 1：1行，-1：自动计算列数（扩展行数）。扩展列数，对比得出布尔值
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # 取前k行，自动计算维度展为一维，
        correct_k = correct[:k].view(-1).float().sum(0)
        # 转换为百分比精度
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()

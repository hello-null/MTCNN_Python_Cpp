from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class PNet(nn.Module):
    def __init__(self):

        super(PNet, self).__init__()

        # suppose we have input with size HxW, then
        # after first layer: H - 2,
        # after pool: ceil((H - 2)/2),
        # after second conv: ceil((H - 2)/2) - 2,
        # after last conv: ceil((H - 2)/2) - 4,
        # and the same for W

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 10, 3, 1)),
                    ("prelu1", nn.PReLU(10)),
                    # 新增 ZeroPad2d 层，手动实现 ceil_mode=True 的边界补足效果，不然ONNX出问题
                    ("pad1", nn.ZeroPad2d((0, 1, 0, 1))),
                    ("pool1", nn.MaxPool2d(2, 2, ceil_mode=False)),
                    ("conv2", nn.Conv2d(10, 16, 3, 1)),
                    ("prelu2", nn.PReLU(16)),
                    ("conv3", nn.Conv2d(16, 32, 3, 1)),
                    ("prelu3", nn.PReLU(32)),
                ]
            )
        )

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        dir_path = path.dirname(__file__)
        weights = np.load(path.join(dir_path, "weights/pnet.npy"), allow_pickle=True)[()]
        state_dict = {k: torch.FloatTensor(v) for k, v in weights.items()}
        self.load_state_dict(state_dict, strict=False)
        # for n, p in self.named_parameters():
        #     p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
            3通道RGB图像，任意大小
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].

        假设：
        - 原始边界框： x1=100, y1=100, x2=200, y2=200 （宽度 w=100，高度 h=100）
        - 预测的偏移量： tx1=-0.1, ty1=-0.1, tx2=0.1, ty2=0.1
        计算校准后的边界框：
            - x1_true = 100 + (-0.1)*100 = 90
            - y1_true = 100 + (-0.1)*100 = 90
            - x2_true = 200 + 0.1*100 = 210
            - y2_true = 200 + 0.1*100 = 210
        结果：边界框从 [100,100,200,200] 扩大为 [90,90,210,210] ，更完整地包围人脸。

            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x) # [B,32,H/2,W/2]
        a = self.conv4_1(x) # [B,2,H/2,W/2] 输入32通道，输出2通道（人脸/非人脸分类）
        b = self.conv4_2(x) # [B,4,H/2,W/2] 输入32通道，输出4通道（边界框回归偏移量）
        a = F.softmax(a, dim=1)
        return b, a


class RNet(nn.Module):
    def __init__(self):

        super(RNet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 28, 3, 1)),
                    ("prelu1", nn.PReLU(28)),
                    # 替换 ceil_mode=True：添加 Padding 层
                    ("pad1", nn.ZeroPad2d((0, 1, 0, 1))),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=False)),
                    ("conv2", nn.Conv2d(28, 48, 3, 1)),
                    ("prelu2", nn.PReLU(48)),
                    # 第二个池化层同样处理
                    ("pad2", nn.ZeroPad2d((0, 1, 0, 1))),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=False)),
                    ("conv3", nn.Conv2d(48, 64, 2, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("flatten", Flatten()),
                    ("conv4", nn.Linear(576, 128)),
                    ("prelu4", nn.PReLU(128)),
                ]
            )
        )

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        dir_path = path.dirname(__file__)
        weights = np.load(path.join(dir_path, "weights/rnet.npy"), allow_pickle=True)[()]
        state_dict = {k: torch.FloatTensor(v) for k, v in weights.items()}
        self.load_state_dict(state_dict, strict=False)
        # weights = np.load(path.join(dir_path, "weights/rnet.npy"), allow_pickle=True)[()]
        # for n, p in self.named_parameters():
        #     p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
            3通道RGB图像，固定大小24x24
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        assert x.shape[2]==x.shape[3] and x.shape[2]==24,'err x'
        x = self.features(x) # B × 128
        a = self.conv5_1(x) # B × 2 输出2个神经元（人脸/非人脸分类）
        b = self.conv5_2(x) # B × 4 输出4个神经元（边界框回归偏移量）
        a = F.softmax(a, dim=1)
        # 边界框回归偏移量，4个元素分别对应(tx1, ty1, tx2, ty2)         
        # 人脸概率，经过softmax处理，2个元素分别对应非人脸和人脸的概率 
        return b, a 


class ONet(nn.Module):
    def __init__(self):

        super(ONet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 3, 1)),
                    ("prelu1", nn.PReLU(32)),
                    ("pad1", nn.ZeroPad2d((0, 1, 0, 1))),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=False)),
                    ("conv2", nn.Conv2d(32, 64, 3, 1)),
                    ("prelu2", nn.PReLU(64)),
                    ("pad2", nn.ZeroPad2d((0, 1, 0, 1))),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=False)),
                    ("conv3", nn.Conv2d(64, 64, 3, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("pad3", nn.ZeroPad2d((0, 1, 0, 1))),
                    ("pool3", nn.MaxPool2d(2, 2, ceil_mode=False)),
                    ("conv4", nn.Conv2d(64, 128, 2, 1)),
                    ("prelu4", nn.PReLU(128)),
                    ("flatten", Flatten()),
                    ("conv5", nn.Linear(1152, 256)),
                    ("drop5", nn.Dropout(0.25)),
                    ("prelu5", nn.PReLU(256)),
                ]
            )
        )

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        dir_path = path.dirname(__file__)
        weights = np.load(path.join(dir_path, "weights/onet.npy"), allow_pickle=True)[()]
        state_dict = {k: torch.FloatTensor(v) for k, v in weights.items()}
        self.load_state_dict(state_dict, strict=False)
        # for n, p in self.named_parameters():
        #     p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
            3通道RGB图像，固定大小48x48
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        assert x.shape[2]==x.shape[3] and x.shape[2]==48,'err x'
        x = self.features(x) # [B,256]
        a = self.conv6_1(x) # [batch_size, 2] 人脸概率，经过softmax处理，2个元素分别对应非人脸和人脸的概率
        b = self.conv6_2(x) # [batch_size, 4] 边界框回归偏移量，4个元素分别对应(tx1, ty1, tx2, ty2)
        c = self.conv6_3(x) # [batch_size, 10] 人脸关键点坐标，10个元素分别对应5个关键点的(x, y)坐标
        a = F.softmax(a, dim=1)
        return c, b, a

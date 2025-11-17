# Commented out IPython magic to ensure Python compatibility.
# import header files
# %matplotlib inline
import torch
import torch.nn as nn
import torchvision
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
import glob
import os
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import time
import copy
import tqdm
import torch
import random
from PIL import Image
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader

# load my google drive
def auth_gdrive():
  from google.colab import drive
  if os.path.exists('content/gdrive/My Drive'): return
  drive.mount('/content/gdrive')
def load_gdrive_dataset():
  loader_assets = 'Tiny_ImageNet.zip'
  auth_gdrive()

# mount my google drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
load_gdrive_dataset()

# unzip dataset
!unzip "/content/gdrive/MyDrive/Tiny_ImageNet.zip"

# define transforms
import torchvision
train_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.Pad(4), torchvision.transforms.RandomCrop(32), torchvision.transforms.ToTensor()])
test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.Pad(4), torchvision.transforms.RandomCrop(32), torchvision.transforms.ToTensor()])

# get data
train_data = torchvision.datasets.ImageFolder("/content/Tiny_ImageNet/train/", transform=train_transforms)
test_data = torchvision.datasets.ImageFolder("/content/Tiny_ImageNet/test/", transform=test_transforms)

# print the number of samples in the training and test sets
classes = [
"n02124075",
"n04067472",
"n04540053",
"n04099969",
"n07749582",
"n01641577",
"n02802426",
"n09246464",
"n07920052",
"n03970156",
"n03891332",
"n02106662",
"n03201208",
"n02279972",
"n02132136",
"n04146614",
"n07873807",
"n02364673",
"n04507155",
"n03854065",
"n03838899",
"n03733131",
"n01443537",
"n07875152",
"n03544143",
"n09428293",
"n03085013",
"n02437312",
"n07614500",
"n03804744",
"n04265275",
"n02963159",
"n02486410",
"n01944390",
"n09256479",
"n02058221",
"n04275548",
"n02321529",
"n02769748",
"n02099712",
"n07695742",
"n02056570",
"n02281406",
"n01774750",
"n02509815",
"n03983396",
"n07753592",
"n04254777",
"n02233338",
"n04008634",
"n02823428",
"n02236044",
"n03393912",
"n07583066",
"n04074963",
"n01629819",
"n09332890",
"n02481823",
"n03902125",
"n03404251",
"n09193705",
"n03637318",
"n04456115",
"n02666196",
"n03796401",
"n02795169",
"n02123045",
"n01855672",
"n01882714",
"n02917067",
"n02988304",
"n04398044",
"n02843684",
"n02423022",
"n02669723",
"n04465501",
"n02165456",
"n03770439",
"n02099601",
"n04486054",
"n02950826",
"n03814639",
"n04259630",
"n03424325",
"n02948072",
"n03179701",
"n03400231",
"n02206856",
"n03160309",
"n01984695",
"n03977966",
"n03584254",
"n04023962",
"n02814860",
"n01910747",
"n04596742",
"n03992509",
"n04133789",
"n03937543",
"n02927161",
"n01945685",
"n02395406",
"n02125311",
"n03126707",
"n04532106",
"n02268443",
"n02977058",
"n07734744",
"n03599486",
"n04562935",
"n03014705",
"n04251144",
"n04356056",
"n02190166",
"n03670208",
"n02002724",
"n02074367",
"n04285008",
"n04560804",
"n04366367",
"n02403003",
"n07615774",
"n04501370",
"n03026506",
"n02906734",
"n01770393",
"n04597913",
"n03930313",
"n04118538",
"n04179913",
"n04311004",
"n02123394",
"n04070727",
"n02793495",
"n02730930",
"n02094433",
"n04371430",
"n04328186",
"n03649909",
"n04417672",
"n03388043",
"n01774384",
"n02837789",
"n07579787",
"n04399382",
"n02791270",
"n03089624",
"n02814533",
"n04149813",
"n07747607",
"n03355925",
"n01983481",
"n04487081",
"n03250847",
"n03255030",
"n02892201",
"n02883205",
"n03100240",
"n02415577",
"n02480495",
"n01698640",
"n01784675",
"n04376876",
"n03444034",
"n01917289",
"n01950731",
"n03042490",
"n07711569",
"n04532670",
"n03763968",
"n07768694",
"n02999410",
"n03617480",
"n06596364",
"n01768244",
"n02410509",
"n03976657",
"n01742172",
"n03980874",
"n02808440",
"n02226429",
"n02231487",
"n02085620",
"n01644900",
"n02129165",
"n02699494",
"n03837869",
"n02815834",
"n07720875",
"n02788148",
"n02909870",
"n03706229",
"n07871810",
"n03447447",
"n02113799",
"n12267677",
"n03662601",
"n02841315",
"n07715103",
"n02504458"

]
print('The number of samples in the training set:', len(train_data))
print('The number of samples in the test set:',len(test_data))

# data loader
import torch
import matplotlib.pyplot as plt
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)
# plot random a batch images
from torchvision.utils import make_grid
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['times'] + plt.rcParams['font.serif']
def show_batch(dl, classes):
  for data, labels in dl:
    fig, ax = plt.subplots(figsize=(32, 16))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(data[:32], nrow=8).squeeze().permute(1, 2, 0).clamp(0,1))
    print('Labels: ', list(map(lambda l: classes[l], labels)))
    break
show_batch(trainloader, classes)

# define the network
import torch.nn as nn
class Flatten(nn.Module):
    def forward(self, x):
        out = x.view(x.size(0), -1)
        return out

# define the Sequential Cognitive Attention Block (SCAB)
import torch.nn as nn
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.Linear1 = nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1, bias=False)
        self.BatchNorm1d = nn.BatchNorm1d(gate_channels // reduction_ratio)
        self.ReLU = nn.ReLU()
        self.Linear2 = nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.Linear2(self.ReLU(self.Linear1(self.avg_pool(x))))
        max_out = self.Linear2(self.ReLU(self.Linear1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=3, number_of_dilation=1, dilation_value=2):# dilation_value=dilation_rate
        super(SpatialGate, self).__init__()
        # the receptive field of dilated convolution with a kernel size of 3 × 3 and a dilation value of 2 is equal to 5 × 5.
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=dilation_value, dilation=dilation_value),
            nn.BatchNorm2d(1),
            nn.ReLU()
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class SCAB(nn.Module):
    def __init__(self, gate_channels):
        super(SCAB, self).__init__()

        self.channel_att = ChannelGate(gate_channels)
        self.spatial_att = SpatialGate()

    def forward(self, x):

        channel_att_map = x * (self.channel_att(x))
        out = x + channel_att_map * (self.spatial_att(channel_att_map))
        return out

"""
Creates a MobileNetV4 Model as defined in:
Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal, Tenghui Zhu, Daniele Moro, Andrew Howard. (2024).
MobileNetV4 - Universal Models for the Mobile Ecosystem
arXiv preprint arXiv:2404.10518.
"""

import torch
import torch.nn as nn
import math


__all__ = ['mobilenetv4_conv_small', 'mobilenetv4_conv_medium', 'mobilenetv4_conv_large',
           'mobilenetv4_hybrid_medium', 'mobilenetv4_hybrid_large']


def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            SCAB(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UniversalInvertedBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 stride,
                 middle_dw_downsample: bool = True,
                 use_layer_scale: bool = False,
                 layer_scale_init_value: float = 1e-5):
        super(UniversalInvertedBottleneck, self).__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size

        if start_dw_kernel_size:
           self.start_dw_conv = nn.Conv2d(in_channels, in_channels, start_dw_kernel_size,
                                          stride if not middle_dw_downsample else 1,
                                          (start_dw_kernel_size - 1) // 2,
                                          groups=in_channels, bias=False)
           self.start_dw_norm = nn.BatchNorm2d(in_channels)

        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = nn.Conv2d(in_channels, expand_channels, 1, 1, bias=False)
        self.expand_norm = nn.BatchNorm2d(expand_channels)
        self.expand_act = nn.ReLU(inplace=True)

        if middle_dw_kernel_size:
           self.middle_dw_conv = nn.Conv2d(expand_channels, expand_channels, middle_dw_kernel_size,
                                           stride if middle_dw_downsample else 1,
                                           (middle_dw_kernel_size - 1) // 2,
                                           groups=expand_channels, bias=False)
           self.middle_dw_norm = nn.BatchNorm2d(expand_channels)
           self.middle_dw_act = nn.ReLU(inplace=True)

        self.proj_conv = nn.Conv2d(expand_channels, out_channels, 1, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(out_channels)

        if use_layer_scale:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

        self.use_layer_scale = use_layer_scale
        self.identity = stride == 1 and in_channels == out_channels

    def forward(self, x):
        shortcut = x

        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_norm(x)

        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_norm(x)
            x = self.middle_dw_act(x)

        x = self.proj_conv(x)
        x = self.proj_norm(x)

        if self.use_layer_scale:
            x = self.gamma * x

        return x + shortcut if self.identity else x


class MobileNetV4(nn.Module):
    def __init__(self, block_specs, num_classes=200):
        super(MobileNetV4, self).__init__()

        c = 3
        layers = []
        for block_type, *block_cfg in block_specs:
            if block_type == 'conv_bn':
                block = ConvBN
                k, s, f = block_cfg
                layers.append(block(c, f, k, s))
            elif block_type == 'uib':
                block = UniversalInvertedBottleneck
                start_k, middle_k, s, f, e = block_cfg
                layers.append(block(c, f, e, start_k, middle_k, s))
            else:
                raise NotImplementedError
            c = f
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_channels = 1280
        self.conv = ConvBN(c, hidden_channels, 1)
        self.classifier = nn.Linear(hidden_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv4_conv_small_SCAB(**kwargs):
    """
    Constructs a MobileNetV4-Conv-Small model
    """
    block_specs = [
        # conv_bn, kernel_size, stride, out_channels
        # uib, start_dw_kernel_size, middle_dw_kernel_size, stride, out_channels, expand_ratio
        # 112px
        ('conv_bn', 3, 2, 32),
        # 56px
        ('conv_bn', 3, 2, 32),
        ('conv_bn', 1, 1, 32),
        # 28px
        ('conv_bn', 3, 2, 96),
        ('conv_bn', 1, 1, 64),
        # 14px
        ('uib', 5, 5, 2, 96, 3.0),  # ExtraDW
        ('uib', 0, 3, 1, 96, 2.0),  # IB
        ('uib', 0, 3, 1, 96, 2.0),  # IB
        ('uib', 0, 3, 1, 96, 2.0),  # IB
        ('uib', 0, 3, 1, 96, 2.0),  # IB
        ('uib', 3, 0, 1, 96, 4.0),  # ConvNext
        # 7px
        ('uib', 3, 3, 2, 128, 6.0),  # ExtraDW
        ('uib', 5, 5, 1, 128, 4.0),  # ExtraDW
        ('uib', 0, 5, 1, 128, 4.0),  # IB
        ('uib', 0, 5, 1, 128, 3.0),  # IB
        ('uib', 0, 3, 1, 128, 4.0),  # IB
        ('uib', 0, 3, 1, 128, 4.0),  # IB
        ('conv_bn', 1, 1, 960),  # Conv
    ]
    return MobileNetV4(block_specs, **kwargs)


def mobilenetv4_conv_medium(**kwargs):
    """
    Constructs a MobileNetV4-Conv-Medium model
    """
    block_specs = [
        ('conv_bn', 3, 2, 32),
        ('conv_bn', 3, 2, 128),
        ('conv_bn', 1, 1, 48),
        # 3rd stage
        ('uib', 3, 5, 2, 80, 4.0),
        ('uib', 3, 3, 1, 80, 2.0),
        # 4th stage
        ('uib', 3, 5, 2, 160, 6.0),
        ('uib', 3, 3, 1, 160, 4.0),
        ('uib', 3, 3, 1, 160, 4.0),
        ('uib', 3, 5, 1, 160, 4.0),
        ('uib', 3, 3, 1, 160, 4.0),
        ('uib', 3, 0, 1, 160, 4.0),
        ('uib', 0, 0, 1, 160, 2.0),
        ('uib', 3, 0, 1, 160, 4.0),
        # 5th stage
        ('uib', 5, 5, 2, 256, 6.0),
        ('uib', 5, 5, 1, 256, 4.0),
        ('uib', 3, 5, 1, 256, 4.0),
        ('uib', 3, 5, 1, 256, 4.0),
        ('uib', 0, 0, 1, 256, 4.0),
        ('uib', 3, 0, 1, 256, 4.0),
        ('uib', 3, 5, 1, 256, 2.0),
        ('uib', 5, 5, 1, 256, 4.0),
        ('uib', 0, 0, 1, 256, 4.0),
        ('uib', 0, 0, 1, 256, 4.0),
        ('uib', 5, 0, 1, 256, 2.0),
        # FC layers
        ('conv_bn', 1, 1, 960),
    ]

    return MobileNetV4(block_specs, **kwargs)


def mobilenetv4_conv_large(**kwargs):
    """
    Constructs a MobileNetV4-Conv-Large model
    """
    block_specs = [
        ('conv_bn', 3, 2, 24),
        ('conv_bn', 3, 2, 96),
        ('conv_bn', 1, 1, 48),
        ('uib', 3, 5, 2, 96, 4.0),
        ('uib', 3, 3, 1, 96, 4.0),
        ('uib', 3, 5, 2, 192, 4.0),
        ('uib', 3, 3, 1, 192, 4.0),
        ('uib', 3, 3, 1, 192, 4.0),
        ('uib', 3, 3, 1, 192, 4.0),
        ('uib', 3, 5, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 3, 0, 1, 192, 4.0),
        ('uib', 5, 5, 2, 512, 4.0),
        ('uib', 5, 5, 1, 512, 4.0),
        ('uib', 5, 5, 1, 512, 4.0),
        ('uib', 5, 5, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 3, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 3, 1, 512, 4.0),
        ('uib', 5, 5, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('conv_bn', 1, 1, 960),
    ]

    return MobileNetV4(block_specs, **kwargs)

# print the model
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenetv4_conv_small_SCAB()
model.to(device)

# print summary of the model
from torchvision import models
from torchsummary import summary
summary(model, (3, 32, 32))

depth = 39
num_classes = 200
epochs = 50
batch_size = 32
batch_size = batch_size
learning_rate = 0.01
momentum = 0.9
l2_param = 0.0001

import torch.optim as optim
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=l2_param)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch // 30))

# training and test processes
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(epochs):
    # training the model
    model.train()
    running_train_loss = 0
    running_validation_loss = 0
    running_test_loss = 0
    running_train_top_1_acc, running_train_top_5_acc = 0, 0
    running_validation_top_1_acc, running_validation_top_5_acc = 0, 0
    running_test_top_1_acc, running_test_top_5_acc = 0, 0


    for train_batch_idx, (input, target) in enumerate(trainloader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)

        # calculate top-1, top-5 accuracy
        top_k = (1, 2)
        max_k = max(top_k)
        _, pred = output.data.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        tmp_acc = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            tmp_acc.append(correct_k.mul_(100.0 / target.size(0)))
        top_1_acc, top_5_acc = tmp_acc[0], tmp_acc[1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss
        running_train_top_1_acc += top_1_acc
        running_train_top_5_acc += top_5_acc

    train_loss = running_train_loss.item() / train_batch_idx
    train_top_1_acc = running_train_top_1_acc.item() / (train_batch_idx + 1)
    train_top_5_acc = running_train_top_5_acc.item() / (train_batch_idx + 1)
    train_loss_list.append(train_loss)
    train_acc_list.append([train_top_1_acc, train_top_5_acc])

    scheduler.step()

    # evaluating trained model
    model.eval()
    with torch.no_grad():
        for test_batch_idx, (input, target) in enumerate(testloader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)

            # calculate top-1, top-5 accuracy
            top_k = (1, 2)
            max_k = max(top_k)
            _, pred = output.data.topk(max_k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            tmp_acc = []
            for k in top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                tmp_acc.append(correct_k.mul_(100.0 / target.size(0)))
            top_1_acc, top_5_acc = tmp_acc[0], tmp_acc[1]

            running_test_loss += loss
            running_test_top_1_acc += top_1_acc
            running_test_top_5_acc += top_5_acc

    test_loss = running_test_loss.item() / test_batch_idx
    test_top_1_acc = running_test_top_1_acc.item() / (test_batch_idx + 1)
    test_top_5_acc = running_test_top_5_acc.item() / (test_batch_idx + 1)
    test_loss_list.append(test_loss)
    test_acc_list.append([test_top_1_acc, test_top_5_acc])

    print("Epoch [{}/{}]".format(epoch + 1, epochs))
    print("train accuracy: {:.2f}%, train loss: {:.4f}%".format(train_top_1_acc, train_loss))
    print("test accuracy: {:.2f}%, test loss: {:.4f}%".format(test_top_1_acc, test_loss))

# plot accuracy and loss curves
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['times'] + plt.rcParams['font.serif']
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.plot(np.array(train_acc_list)[:, 0], lw=2, color = 'green', label = 'Training Accuracy')
#ax1.plot(np.array(validation_acc_list)[:, 0], lw=2, color = 'blue', label = 'Validation Accuracy')
ax1.plot(np.array(test_acc_list)[:, 0], lw=2, color = 'cyan', label = 'Test Accuracy')
ax2 = ax1.twinx()
ax2.plot(train_loss_list, lw=2, color = 'red',  label = 'Training Loss')
#ax2.plot(validation_loss_list, lw=2, color = 'deepskyblue',  label = 'Validation Loss')
ax2.plot(test_loss_list, lw=2, color = 'violet', label = 'Test Loss')
fig.legend(loc="center", fontsize=10)
ax1.set_xlabel('Epochs', labelpad=10, fontweight='bold')
ax1.set_ylabel('Accuracy', labelpad=10, fontweight='bold')
ax2.set_ylabel('Loss', labelpad=10, fontweight='bold')
plt.show()

model.eval()

for test_batch_idx, (input, target) in enumerate(testloader):

    if test_batch_idx != 0:
        break

    input, target = input.to(device), target.to(device)
    output = model(input)
    loss = criterion(output, target)

    # calculate top-1, top-5 accuracy
    tmp_prd = output.data
    top_k = (1, 2)
    max_k >= max(top_k)
    _, pred = output.data.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    tmp_acc = []
    for k in top_k:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        tmp_acc.append(correct_k.mul_(100.0 / target.size(0)))
    top_1_acc, top_5_acc = tmp_acc[0].item(), tmp_acc[1].item()
    print("Top-1 accuracy: {:.2f}, Top-5 accuracy: {:.2f}".format(top_1_acc, top_5_acc))
    tmp_img, tmp_prd, tmp_lbl = input, pred, target

# plot sample predictions in test dataset
import seaborn as sns
sns.set(style="white", font_scale=1)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['times'] + plt.rcParams['font.serif']
fig, ax = plt.subplots(4, 4, figsize=(20, 20))
ax = ax.flat
rand_idx = np.random.choice(np.arange(target.size(0)), 16, replace=False)

for i in range(len(ax)):
    tmp_idx = rand_idx[i]
    ax[i].imshow(tmp_img[tmp_idx].permute(1, 2, 0).detach().cpu().numpy())
    ax[i].set_title("True Label: {}, Predicted Label: {}".format(
        classes[tmp_lbl[tmp_idx]], classes[tmp_prd[0, tmp_idx]]), pad=12, color='black')
    ax[i].axis("off")

# plot test confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
y_pred = []
y_true = []
# iterate over test data
for inputs, labels in testloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        output = model(inputs) # feed network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # save prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # save truth
cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot(cmap=plt.cm.Greys)

# calflops: a FLOPs and Params calculate tool for neural networks
# https://github.com/MrYxJ/calculate-flops.pytorch?tab=readme-ov-file
!pip install --upgrade calflops
# !pip install calflops-*-py3-none-any.whl

from calflops import calculate_flops
from torchvision import models
model = mobilenetv4_conv_small_SCAB()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch_size = 32
input_shape = (batch_size, 3, 32, 32)
flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=200)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))




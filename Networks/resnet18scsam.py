# print the Python version
import sys
print("Python version:", (sys.version))

# print the Pythorch version
import torch;
print("Pythorch version:", (torch.__version__))

# assign GPU
import os
GPU = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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
import os
drive.mount('/content/gdrive', force_remount=True)
load_gdrive_dataset()

# unzip Fonts
!unzip "/content/gdrive/MyDrive/Fonts.zip"

# install time new roman or times font
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
# path to the times font file
font_path = Path("/content/Fonts/times.ttf")
# load and register the times font
times_font = fm.FontProperties(fname=font_path)
# add the font to Matplotlib's font manager
fm.fontManager.addfont(font_path)
# set the global font to times
plt.rcParams['font.family'] = times_font.get_name()

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

def conv_3x3(in_planes, out_planes, stride=1):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return conv

class BasicBlock(nn.Module):
    expansion = 1 # dilation

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv_1 = conv_3x3(in_planes, out_planes, stride)
        self.bn_1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv_3x3(out_planes, out_planes)
        self.bn_2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.downsample != None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4 # dilation

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv_1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_planes)
        self.conv_2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn_2 = nn.BatchNorm2d(out_planes)
        self.conv_3 = nn.Conv2d(out_planes, out_planes * 4, kernel_size=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(out_planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.bn_3(out)

        if self.downsample != None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes, att_type=None):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        if network_type == "Tiny_ImageNet":
            self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(3)
        elif network_type == "ImageNet":
            self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type == "SCAB":
            self.scab_1 = SCAB(64 * block.expansion)
            self.scab_2 = SCAB(128 * block.expansion)
            self.scab_3 = SCAB(256 * block.expansion)
        else:
            self.scab_1, self.scab_2, self.scab_3 = None, None, None

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        nn.init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split("."[-1]) == "weight":
                if "conv" in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode="fan_out")
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(
            self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.layer1(x)
        if self.scab_1 != None:
            x = self.scab_1(x)

        x = self.layer2(x)
        if self.scab_2 != None:
            x = self.scab_2(x)

        x = self.layer3(x)
        if self.scab_3 != None:
            x = self.scab_3(x)

        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResidualNet(network_type, depth, num_classes, att_type):
    assert network_type in ["Tiny_ImageNet"]
    assert depth in [18, 34, 50, 101], "network depth should be 18, 34, 50 or 101"

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)
    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)
    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)
    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model

depth = 18
num_classes = 200
epochs = 50
batch_size = 32
batch_size = batch_size
learning_rate = 0.01
momentum = 0.9
l2_param = 0.0001

import torch.optim as optim
model = ResidualNet("Tiny_ImageNet", depth, num_classes, "SCAB").to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=l2_param)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch // 30))

# print the network
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# print summary of the network
from torchvision import models
from torchsummary import summary
import torch.nn.functional as F
summary(model, (3, 32, 32))

# training and test processes
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(epochs):
    # training the model
    model.train()
    running_train_loss = 0
    running_test_loss = 0
    running_train_top_1_acc, running_train_top_5_acc = 0, 0
    running_test_top_1_acc, running_test_top_5_acc = 0, 0


    for train_batch_idx, (input, target) in enumerate(trainloader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)

        # calculate top-1, top-5 accuracy
        top_k = (1, 5)
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

    # test the model
    model.eval()
    with torch.no_grad():
        for test_batch_idx, (input, target) in enumerate(testloader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)

            # calculate top-1, top-5 accuracy
            top_k = (1, 5)
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
    print("train loss: {:.4f}, train top-1: {:.2f}%, train top-5: {:.2f}%".format(
        train_loss, train_top_1_acc, train_top_5_acc))
    print("test loss: {:.4f}, test top-1: {:.2f}%, test top-5: {:.2f}%".format(
        test_loss, test_top_1_acc, test_top_5_acc))

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
fig.legend(loc="center",     bbox_to_anchor=(0.75, 0.5), fontsize=8)
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
    top_k = (1, 5)
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

model = ResidualNet("Tiny_ImageNet", depth, num_classes, "SCAB").to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch_size = 32
input_shape = (batch_size, 3, 32, 32)
flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=200)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))




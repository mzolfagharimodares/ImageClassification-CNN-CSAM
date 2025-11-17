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
  loader_assets = 'CIFAR10.zip'
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
!unzip "/content/gdrive/MyDrive/CIFAR10.zip"

# define transforms
import torchvision
train_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.Pad(4), torchvision.transforms.RandomCrop(32), torchvision.transforms.ToTensor()])
test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.Pad(4), torchvision.transforms.RandomCrop(32), torchvision.transforms.ToTensor()])

# get data
train_data = torchvision.datasets.ImageFolder("/content/CIFAR10/train/", transform=train_transforms)
test_data = torchvision.datasets.ImageFolder("/content/CIFAR10/test/", transform=test_transforms)

# print the number of samples in the training, validation, and test sets
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('The number of samples in the training set:', len(train_data))
print('The number of samples in the test set:',len(test_data))

# check sample data in the dataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font="times")
sns.set(style="white", font_scale=1)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['times'] + plt.rcParams['font.serif']
fig, ax = plt.subplots(4, 4, figsize=(20, 18))
ax = ax.flat
for i in range(len(ax)):
    smp_idx = torch.randint(len(test_data), size=(1,)).item()
    img, lbl = test_data[smp_idx]
    ax[i].imshow(img.permute(1, 2, 0))
    ax[i].set_title("Class: {}".format(classes[lbl]), pad=12, color='black')
    ax[i].axis("off")

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

# define the Parallel Cognitive Attention Block (PCAB)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.BatchNorm1d(gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))
        max_pool = F.max_pool2d(x, x.size(2), stride=x.size(2))
        out = self.mlp(avg_pool + max_pool)
        out = out.unsqueeze(2).unsqueeze(3).expand_as(x)
        return out
class SpatialGate(nn.Module):
    def __init__(self, gate_channels, number_of_dilation=1, dilation_value=2): # dilation_value=dilation_rate
        super(SpatialGate, self).__init__()
        self.gate_channels = gate_channels
        self.reduced_channel = gate_channels
        # the receptive field of dilated convolution with a kernel size of 3 × 3 and a dilation value of 2 is equal to 5 × 5.
        self.conv = nn.Sequential(
            nn.Conv2d(gate_channels, self.reduced_channel, kernel_size=3, padding=dilation_value, dilation=dilation_value),
            nn.BatchNorm2d(self.reduced_channel),
            nn.ReLU()
            )

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))
        max_pool = F.max_pool2d(x, x.size(2), stride=x.size(2))
        out = self.conv(avg_pool + max_pool)
        out = self.conv(x).expand_as(x)
        return out
class PCAB(nn.Module):
    def __init__(self, gate_channels):
        super(PCAB, self).__init__()
        self.channel_att = ChannelGate(gate_channels)
        self.spatial_att = SpatialGate(gate_channels)

    def forward(self, x):
        att_map = torch.sigmoid(torch.add(self.channel_att(x), self.spatial_att(x)))
        out = x + att_map * x
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
        if network_type == "CIFAR10":
            self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(3)
        elif network_type == "ImageNet":
            self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type == "PCAB":
            self.pcab_1 = PCAB(64 * block.expansion)
            self.pcab_2 = PCAB(128 * block.expansion)
            self.pcab_3 = PCAB(256 * block.expansion)
        else:
            self.pcab_1, self.pcab_2, self.pcab_3 = None, None, None

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
        if self.pcab_1 != None:
            x = self.pcab_1(x)

        x = self.layer2(x)
        if self.pcab_2 != None:
            x = self.pcab_2(x)

        x = self.layer3(x)
        if self.pcab_3 != None:
            x = self.pcab_3(x)

        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResidualNet(network_type, depth, num_classes, att_type):
    assert network_type in ["CIFAR10"]
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
num_classes = 10
epochs = 50
batch_size = 32
batch_size = batch_size
learning_rate = 0.01
momentum = 0.9
l2_param = 0.0001

import torch.optim as optim
model = ResidualNet("CIFAR10", depth, num_classes, "PCAB").to(device)
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

!pip install grad-cam

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# plot an image from the training set
import tensorflow
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
path = load_img('/content/CIFAR10/train/dog/blenheim_spaniel_s_000006.png')
data = img_to_array(path)
samples = np.expand_dims(data, 0)
plt.imshow(path)
plt.show()

# importing Image class from PIL package
from PIL import Image
path1 = ('/content/CIFAR10/train/dog/blenheim_spaniel_s_000006.png')
Image.open(path1).convert('RGB')

# pick up layers for visualization
target_layers = [model.layer4[-1]]

rgb_img = Image.open(path1).convert('RGB')
# max min normalization
rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
# create an input tensor image for your model
input_tensor = torchvision.transforms.functional.to_tensor(rgb_img).unsqueeze(0).float()
# construct the CAM object once, and then re-use it on many images:
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
# in this example grayscale_cam has only one image in the batch:
grayscale_cam = cam(input_tensor=input_tensor)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# plot GradCAM of image
Image.fromarray(visualization, 'RGB')

cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# plot GradCAMPlusPlus of image
Image.fromarray(visualization, 'RGB')

cam = ScoreCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# plot ScoreCAM of image
Image.fromarray(visualization, 'RGB')

# calflops: a FLOPs and Params calculate tool for neural networks
# https://github.com/MrYxJ/calculate-flops.pytorch?tab=readme-ov-file
!pip install --upgrade calflops
# !pip install calflops-*-py3-none-any.whl

from calflops import calculate_flops
from torchvision import models

model = ResidualNet("CIFAR10", depth, num_classes, "PCAB").to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch_size = 32
input_shape = (batch_size, 3, 32, 32)
flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=10)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))




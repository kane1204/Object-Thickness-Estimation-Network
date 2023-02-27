from collections import OrderedDict
import math
import torch
import torch.nn as nn
import collections
import torchvision
# https://medium.com/artificialis/going-deep-an-introduction-to-depth-estimation-with-fully-convolutional-residual-networks-2501f3be86b9

class FasterUpConv(nn.Module):

    def __init__(self, in_channels):
        super(FasterUpConv, self).__init__()

        self.conv1_ = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, in_channels // 2, kernel_size=3)),
            ('bn1', nn.BatchNorm2d(in_channels // 2)),
        ]))

        self.conv2_ = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, in_channels // 2, kernel_size=(2, 3))),
            ('bn1', nn.BatchNorm2d(in_channels // 2)),
        ]))

        self.conv3_ = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, in_channels // 2, kernel_size=(3, 2))),
            ('bn1', nn.BatchNorm2d(in_channels // 2)),
        ]))

        self.conv4_ = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, in_channels // 2, kernel_size=2)),
            ('bn1', nn.BatchNorm2d(in_channels // 2)),
        ]))

        self.ps = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1_(nn.functional.pad(x, (1, 1, 1, 1)))
        x2 = self.conv2_(nn.functional.pad(x, (1, 1, 0, 1)))
        x3 = self.conv3_(nn.functional.pad(x, (0, 1, 1, 1)))
        x4 = self.conv4_(nn.functional.pad(x, (0, 1, 0, 1)))

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.ps(x)
        return x
class FasterUpProjModule(nn.Module):
    def __init__(self, in_channels):
        super(FasterUpProjModule, self).__init__()
        out_channels = in_channels // 2

        self.upper_branch = nn.Sequential(collections.OrderedDict([
            ('faster_upconv', FasterUpConv(in_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(out_channels, out_channels,
             kernel_size=3, stride=1, padding=1, bias=False)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
        ]))
        self.bottom_branch = FasterUpConv(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.upper_branch(x)
        x2 = self.bottom_branch(x)
        x = x1 + x2
        x = self.relu(x)
        return x

class FasterUpProj(nn.Module):
    def __init__(self, in_channel):
        super(FasterUpProj, self).__init__()

        self.layer1 = FasterUpProjModule(in_channel)
        self.layer2 = FasterUpProjModule(in_channel // 2)
        self.layer3 = FasterUpProjModule(in_channel // 4)
        self.layer4 = FasterUpProjModule(in_channel // 8)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def weights_init(m):
    """
    Initializes the weights of the convolutional and batch normalization layers in a neural network.

    Args:
        m (nn.Module): A neural network module.

    Returns:
        None

    Notes:
        The function initializes the weights of the convolutional and batch normalization layers in a neural network
        with random values drawn from a normal distribution with a mean of zero and a variance of 2/n, where n is the 
        number of input neurons to the layer. If the layer has a bias term, it is initialized to zero. 
    """
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class ResNet(nn.Module):
    def __init__(self, layers=18, output_size=(128, 128), in_channels=3, pretrained=True):
        super(ResNet, self).__init__()

        pretrained_model = torchvision.models.__dict__[f'resnet{layers}'](pretrained=pretrained)
        
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.output_size = output_size
        self.relu = pretrained_model.relu
        self.maxpool = pretrained_model.maxpool
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4

        del pretrained_model #free memory

        num_channels = 512 if layers <= 34 else 2048

        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels // 2)
        self.upsample = FasterUpProj(num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.upsample.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.conv2(x4)
        x = self.bn2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch1 = nn.BatchNorm2d(64)
        self.res1 = ResBlock(64, 64, 1)
        self.res2 = ResBlock(64, 128, 2)
        self.res3 = ResBlock(128, 256, 2)
        self.res4 = ResBlock(256, 512, 2)
        self.glob = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Linear(512, 2)
        #self.siggi = nn.Sigmoid()

    def forward(self, rein):
        x = F.max_pool2d(F.relu(self.conv1(rein)), (3, 3), stride=2)
        x = self.batch1(x)
        x = self.res1.forward(x)
        x = self.res2.forward(x)
        x = self.res3.forward(x)
        x = self.res4.forward(x)
        x = self.glob(x)
        x = torch.flatten(x, start_dim=1)
        x = self.flat(x)
        x = F.sigmoid(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,in_channel, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channels, kernel_size = 3, stride=stride, padding=1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.addi = nn.Conv2d(in_channel, out_channels, kernel_size = 1, stride=stride)
        self.batch3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        addi = self.addi(x)
        addi = self.batch3(addi)
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = F.relu(self.conv2(x))
        x = self.batch1(x)
        x = x+addi
        return x
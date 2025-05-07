import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)


    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)


    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)


    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))


    return sobel_x, sobel_y



def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input



class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        y = F.silu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y



class ESAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ESAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.ban = nn.BatchNorm2d(out_channels)
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, in_channels)


    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, x)
        y = F.silu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = self.conv2(y)
        y = F.silu(self.ban(y))

        return y



class Edgenet(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Edgenet, self).__init__()
        self.down1 = Downsample_block(in_chan, 32)
        self.down2 = Downsample_block(32, 64)
        self.down3 = Downsample_block(64, 128)
        self.down4 = Downsample_block(128, 256)
        self.down5 = Downsample_block(256, 512)

        self.center = ESAM(512, 512)

        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.esam5 = ESAM(512, 256)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.esam4 = ESAM(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.esam3 = ESAM(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.esam2 = ESAM(64, 32)
        self.up1 = nn.ConvTranspose2d(32, out_chan, 2, stride=2)


    def forward(self, x):
        x1, y1 = self.down1(x)
        x2, y2 = self.down2(x1)
        x3, y3 = self.down3(x2)
        x4, y4 = self.down4(x3)
        x5, y5 = self.down5(x4)

        x = self.center(x5)

        x = self.up5(x)
        x = torch.cat([x, y5], dim=1)
        x = self.esam5(x)

        x = self.up4(x)
        x = torch.cat([x, y4], dim=1)
        x = self.esam4(x)

        x = self.up3(x)
        x = torch.cat([x, y3], dim=1)
        x = self.esam3(x)

        x = self.up2(x)
        x = torch.cat([x, y2], dim=1)
        x = self.esam2(x)

        x = self.up1(x)

        return x
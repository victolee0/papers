import torch
import torch.nn as nn
import math


def conv2x(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(out_features, out_features, kernel_size=3),
        nn.ReLU()
    )

def copy_crop(down, up):
    diff = math.ceil((down.shape[-1] - up.shape[-1])/2)
    down = down[:, :, diff:down.shape[-2] - diff, diff:down.shape[-1] - diff]
    up = torch.concat([down, up], 1)
    return up


class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super(UNet, self).__init__()
        
        
        self.down1 = nn.Sequential(
            conv2x(1, 64)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv2x(64, 128),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv2x(128, 256),
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv2x(256, 512)
        )
        
        self.intermediate = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv2x(512, 1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        )
        
        self.up1 = nn.Sequential(
            conv2x(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )
        self.up2 = nn.Sequential(
            conv2x(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )
        self.up3 = nn.Sequential(
            conv2x(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )
        self.up4 = conv2x(128, 64)

        self.conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        inter = self.intermediate(down4)
        
        
        inter = copy_crop(down4, inter)
        up1 = self.up1(inter)
        
        up2 = copy_crop(down3, up1)
        up2 = self.up2(up2)
        
        up3 = copy_crop(down2, up2)
        up3 = self.up3(up3)
        
        up4 = copy_crop(down1, up3)
        up4 = self.up4(up4)
        
        out = self.conv(up4)
        
        return out
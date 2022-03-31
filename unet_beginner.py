import torch
import torch.nn as nn
import math

class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super(UNet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.intermediate = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        )
        
        self.up1 = nn.Sequential(
            
            nn.Conv2d(1024, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(self.maxpool(down1))
        down3 = self.down3(self.maxpool(down2))
        down4 = self.down4(self.maxpool(down3))
        
        inter = self.intermediate(self.maxpool(down4))
        
        diff = math.ceil((down4.shape[-1] - inter.shape[-1])/2)
        down4 = down4[:, :, diff:down4.shape[-2] - diff, diff:down4.shape[-1] - diff]
        
        inter = torch.concat([down4, inter], 1)

        up1 = self.up1(inter)
        
        diff = math.ceil((down3.shape[-1] - up1.shape[-1])/2)
        down3 = down3[:, :, diff:down3.shape[-2] - diff, diff:down3.shape[-1] - diff]

        up2 = torch.concat([down3, up1], 1)
        up2 = self.up2(up2)
        
        diff = math.ceil((down2.shape[-1] - up2.shape[-1])/2)
        down2 = down2[:, :, diff:down2.shape[-2] - diff, diff:down2.shape[-1] - diff]
        up3 = torch.concat([down2, up2], 1)
        up3 = self.up3(up3)
        
        diff = math.ceil((down1.shape[-1] - up3.shape[-1])/2)
        down1 = down1[:, :, diff:down1.shape[-2] - diff, diff:down1.shape[-1] - diff]
        up4 = torch.concat([down1, up3], 1)
        up4 = self.up4(up4)
        
        out = self.conv(up4)
        
        return out
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))

class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU()
        )
        self.c2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

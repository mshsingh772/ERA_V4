import torch.nn as nn
import torch.nn.functional as F
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False) #28
        self.norm1 = nn.BatchNorm2d(8)
        self.drop = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1, bias=False) #28
        self.norm2 = nn.BatchNorm2d(12)
        self.drop = nn.Dropout2d(0.1)
        self.pool = nn.MaxPool2d(2, 2) #14

        self.conv3 = nn.Conv2d(12, 16, 3, padding=1, bias=False) #14
        self.norm3 = nn.BatchNorm2d(16)
        self.drop = nn.Dropout2d(0.1)
        self.conv4 = nn.Conv2d(16, 20, 3, padding=1, bias=False) #14
        self.norm4 = nn.BatchNorm2d(20)
        self.drop = nn.Dropout2d(0.1)
        self.pool = nn.MaxPool2d(2,2) #7

        self.conv5 = nn.Conv2d(20, 24, 3, bias=False) #5
        self.norm5 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24, 28, 3, bias=False) #3
        self.antman = nn.Conv2d(28, 10 , 1, bias=False)#3
        self.gap = nn.AvgPool2d(3)#1

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = self.norm1(x)
      x = self.drop(x)
      x = F.relu(self.conv2(x))
      x = self.norm2(x)
      x = self.drop(x)
      x = self.pool(x)

      x = F.relu(self.conv3(x))
      x = self.norm3(x)
      x = self.drop(x)
      x = F.relu(self.conv4(x))
      x = self.norm4(x)
      x = self.drop(x)
      x = self.pool(x)

      x = F.relu(self.conv5(x))
      x = self.norm5(x)
      x = F.relu(self.conv6(x))
      x = self.antman(x)
      x = self.gap(x)
      x = x.view(-1, 10)

      return F.log_softmax(x, dim=1)
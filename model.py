import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False),   # r_in = 1, jin = 1 jout=1*2 =2, k=3, rout = 1+(3-1)*1 = 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, stride=1, dilation=2, bias=False),# r_in = 3, jin = 2 jout=2*1 =2, k=5, rout = 3+(5-1)*2= 11
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(64)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, stride=2, bias=False), # r_in = 11, jin = 2 jout=2*2 =4, k=3, rout = 11+(3-1)*2 = 15
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, stride=1, bias=False),# r_in = 15, jin = 4 jout=4*1 =4, k=3, rout = 15+(3-1)*4 = 23
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(128)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=0, stride=2, bias=False),# r_in = 23, jin = 4 jout=4*2 =8, k=3, rout = 23+(3-1)*4 = 31
            # r_in = 21, jin = 8 jout=8*1 =8, k=1, rout = 5+(1-1)*8 = 21
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(128)
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), padding=0, stride=1, bias=False),# r_in = 31, jin = 8 jout=1*4 =8, k=1, rout = 31+(1-1)*1 = 21
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3))  # r_in = 31, jin = 8,s=3, jout=1*8 =8, k=3, rout = 31+(3-1)8 = 47
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        #print(x.shape)
        x = self.gap(x)
        #print(x.shape)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=-1)
        return x

"""Small VGG6 variant for a new <1MB CNN checkpoint. Not compatible with large checkpoint."""
import torch
from torch import nn
class SmallVGG6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,16,3,padding=1); self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(16,16,3,padding=1); self.bn2=nn.BatchNorm2d(16)
        self.conv3=nn.Conv2d(16,32,3,padding=1); self.bn3=nn.BatchNorm2d(32)
        self.conv4=nn.Conv2d(32,32,3,padding=1); self.bn4=nn.BatchNorm2d(32)
        self.conv5=nn.Conv2d(32,64,3,padding=1); self.bn5=nn.BatchNorm2d(64)
        self.conv6=nn.Conv2d(64,64,3,padding=1); self.bn6=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2)
        # equivalent to AdaptiveAvgPool2d((1,1)) after the third maxpool
        self.gap=nn.AvgPool2d(4)
        self.flatten=nn.Flatten()
        self.fc=nn.Linear(64,10)
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.pool(x)
        x=self.relu(self.bn3(self.conv3(x)))
        x=self.relu(self.bn4(self.conv4(x)))
        x=self.pool(x)
        x=self.relu(self.bn5(self.conv5(x)))
        x=self.relu(self.bn6(self.conv6(x)))
        x=self.pool(x)
        return self.fc(self.flatten(self.gap(x)))

if __name__=='__main__':
    m=SmallVGG6()
    n=sum(p.numel() for p in m.parameters())
    print('Trainable parameters:',n,'FP32 parameter bytes:',n*4)
    print('Output shape:',tuple(m(torch.zeros(1,3,32,32)).shape))

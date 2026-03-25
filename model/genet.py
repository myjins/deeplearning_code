import torch
import torch.nn as nn
import torch.nn.functional as F

class GatherExcite(nn.Module):
    def __init__(self,channels,extent=0):
        super().__init__()

        self.extent=extent

        if extent==0:
            self.gather=nn.AdaptiveAvgPool2d(1)
        else:
            self.gather=nn.Conv2d(channels,channels,kernel_size=extent,
                                  stride=extent,groups=channels,bias=False)

        self.excite=nn.Sequential(nn.Conv2d(channels,channels,1),nn.Sigmoid())

    def forward(self,x):
        context=self.gather()
        weights=self.excite(context)

        if self.extent!=0:
            weights=F.interpolate(
                weights,
                size=x.shape[2:],
                mode='nearest'
            )

        return x*weights

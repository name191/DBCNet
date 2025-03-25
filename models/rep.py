import torch
from torch import nn

from models.encoders.crossnet import SEBlock
class FFM(nn.Module):
    def __init__(self,inc):
        super().__init__()
        self.conv_act=nn.Sequential(
            nn.Conv2d(2*inc,inc,3,1,1),
            nn.GELU()
        )
        self.ca=SEBlock('avg',inc,2)
        self.attforcross=SEBlock('avg',inc,2)
        self.attforvssm=SEBlock('avg',inc,2)
    def forward(self,x_vssm,x_cross):
        x=torch.cat([x_vssm,x_cross],dim=1)
        x=self.conv_act(x)
        x=self.ca(x)
        res_cross=self.attforcross(x_cross)
        res_vssm=self.attforvssm(x_vssm)
        out=x+res_vssm+res_cross
        return out

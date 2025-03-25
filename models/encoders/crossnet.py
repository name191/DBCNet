import torch
from torch import nn
class SEBlock(nn.Module):
    def __init__(self, mode, channels, ratio):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v
class crossblock(nn.Module):
    def __init__(self, inc, outc, l=9,ifse=True,norm="gn",group=4):
        super(crossblock, self).__init__()
        self.ifse=ifse
        if norm=="gn":
            self.right1 = nn.Sequential(
                nn.Conv2d(inc, outc // 2, kernel_size=(l, 1), stride=1, padding=(l // 2, 0)),
                nn.GroupNorm(outc // (group*2), outc // 2),
                nn.ReLU(),

            )
            self.right2 = nn.Sequential(
                nn.Conv2d(inc, outc // 2, kernel_size=(1, l), stride=1, padding=(0, l // 2)),
                nn.GroupNorm(outc // (group*2), outc // 2),
                nn.ReLU()
            )
            self.left = nn.Sequential(
                nn.Conv2d(inc, outc // 2, 3, padding=1),
                nn.GroupNorm(outc // (group*2), outc // 2),
                nn.ReLU(inplace=True)
            )
            self.conv = nn.Sequential(
                nn.Conv2d(3 * outc // 2, outc, 3, padding=1),
                nn.GroupNorm(outc // group, outc),
                nn.ReLU(inplace=True)
            )
        elif norm == "bn":
            self.right1 = nn.Sequential(
                nn.Conv2d(inc, outc // 2, kernel_size=(l, 1), stride=1, padding=(l // 2, 0)),
                nn.BatchNorm2d(outc//2),
                nn.ReLU(),

            )
            self.right2 = nn.Sequential(
                nn.Conv2d(inc, outc // 2, kernel_size=(1, l), stride=1, padding=(0, l // 2)),
                nn.BatchNorm2d(outc//2),
                nn.ReLU()
            )
            self.left = nn.Sequential(
                nn.Conv2d(inc, outc // 2, 3, padding=1),
                nn.BatchNorm2d(outc//2),
                nn.ReLU(inplace=True)
            )
            self.conv = nn.Sequential(
                nn.Conv2d(3 * outc // 2, outc, 3, padding=1),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True)
            )

        self.relu = nn.ReLU()
        self.se = SEBlock("avg", 3*outc//2, 2)

    def forward(self, x):
        left = self.left(x)
        right1 = self.right1(x)
        right2 = self.right2(x)
        if self.ifse:
            out = self.se(torch.cat((left,right1, right2), dim=1))
        else:
            out=torch.cat((left,right1, right2), dim=1)
        out = self.conv(out)
        return out

class CrossNet(nn.Module):
    def __init__(self,layers,dim,norm='bn',inchannel=3,l=9):
        self.depth=layers
        self.dim=dim
        super().__init__()
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            if i==0:
                self.layers.append(crossblock(inc=inchannel,outc=dim*(2**i),l=l,norm=norm,ifse=False))
            elif i==1:
                self.layers.append(crossblock(inc=dim*(2**(i-1)),outc=dim*(2**i),l=l,norm=norm,ifse=False))
            else:
                self.layers.append(crossblock(inc=dim*(2**(i-1)),outc=dim*(2**i),l=l,norm=norm))
        self.pool=nn.MaxPool2d(2,2)
    def forward(self,x):
        out=[]
        for i,layer in enumerate(self.layers):
            x=layer(x)
            out.append(x)
            if i<self.depth-1:
                x=self.pool(x)
        return out


if __name__=="__main__":
    model=crossencoder(6,16)
    x=torch.randn([2,3,512,512])
    outlist=model(x)
    for out in outlist:
        print(out.shape)

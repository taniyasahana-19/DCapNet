import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_ch
        for f in features:
            self.encs.append(DoubleConv(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f
        self.bottleneck = DoubleConv(prev, prev*2)
        self.upconvs = nn.ModuleList()
        self.decs = nn.ModuleList()
        rev = list(reversed(features))
        decoder_prev = prev*2
        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(decoder_prev, f, kernel_size=2, stride=2))
            self.decs.append(DoubleConv(decoder_prev, f))
            decoder_prev = f
        self.final = nn.Conv2d(decoder_prev, out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encs, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decs, reversed(skips)):
            x = up(x)
            if x.size() != skip.size():
                x = TF.resize(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        x = self.final(x)
        return torch.sigmoid(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class BandReweight(nn.Module):
    def __init__(self, bands: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, bands // reduction)
        self.fc1 = nn.Linear(bands, hidden)
        self.fc2 = nn.Linear(hidden, bands)

    def forward(self, x, return_weights: bool = False):
        s = x.mean(dim=(2, 3))  # (B,C)
        w = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))  # (B,C)
        out = x * w.unsqueeze(-1).unsqueeze(-1)
        if return_weights:
            return out, w
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=12, base=16, band_reweight=False, band_drop_p=0.0):
        super().__init__()
        self.band_reweight = BandReweight(in_ch) if band_reweight else nn.Identity()
        # Drop entire channels at input (band dropping). This matches “channel dropout” behavior. [web:151]
        self.band_drop = nn.Dropout2d(p=band_drop_p) if band_drop_p > 0 else nn.Identity()

        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # store weights for logging (only when using BandReweight)
        if hasattr(self.band_reweight, "forward") and not isinstance(self.band_reweight, nn.Identity):
            x, w = self.band_reweight(x, return_weights=True)
            self.last_band_weights = w.detach()  # (B,C)
        else:
            x = self.band_reweight(x)
            self.last_band_weights = None

        x = self.band_drop(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

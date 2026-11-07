"""Compact U-Net ISP models: LearnedISP (packed RAW -> RGB) and HybridISP (classical front-end + residual U-Net)."""
import torch, torch.nn as nn, torch.nn.functional as F


def conv(cin, cout, sep):
    if sep and cin > 8:   # depthwise-separable conv: ~8x fewer MACs
        return nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1, groups=cin), nn.Conv2d(cin, cout, 1))
    return nn.Conv2d(cin, cout, 3, padding=1)


class Block(nn.Module):
    def __init__(self, cin, cout, sep):
        super().__init__()
        self.f = nn.Sequential(conv(cin, cout, sep), nn.LeakyReLU(0.1, True), conv(cout, cout, sep), nn.LeakyReLU(0.1, True))
    def forward(self, x): return self.f(x)


class UNet(nn.Module):
    def __init__(self, cin, cout, base=16, depth=3, sep=True):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([Block(cin, ch[0], sep)] + [Block(ch[i], ch[i + 1], sep) for i in range(depth)])
        self.dec = nn.ModuleList([Block(ch[i + 1] + ch[i], ch[i], sep) for i in reversed(range(depth))])
        self.out = nn.Conv2d(ch[0], cout, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2)); skips.append(x)
        x = skips[-1]
        for j, d in enumerate(self.dec):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = d(torch.cat([x, skips[-2 - j]], 1))
        return self.out(x)


def _sigma_map(sigma, ref):                 # (B,) -> (B,1,H,W)
    return sigma.view(-1, 1, 1, 1).expand(-1, 1, ref.shape[2], ref.shape[3])


class LearnedISP(nn.Module):
    """Pure learned ISP. Input: packed RGGB (B,4,H/2,W/2) + noise estimate. Output RGB at full res (pixel shuffle).
    Global channel means are fed as extra channels so the net can infer the illuminant (white balance)."""
    def __init__(self, base=16, depth=3, sep=True):
        super().__init__()
        self.net = UNet(4 + 4 + 1, 12, base, depth, sep)
    def forward(self, packed, sigma):
        ctx = packed.mean((2, 3), keepdim=True).expand(-1, -1, packed.shape[2], packed.shape[3])
        y = self.net(torch.cat([packed, ctx, _sigma_map(sigma, packed)], 1))
        return F.pixel_shuffle(y, 2)


class HybridISP(nn.Module):
    """Hybrid: classical WB/demosaic/CCM/gamma/tone front-end output + noise level -> residual U-Net.
    Network only has to learn the *correction* (denoise, detail recovery, colour/WB error fix)."""
    def __init__(self, base=16, depth=3, sep=True):
        super().__init__()
        self.net = UNet(3 + 1, 3, base, depth, sep)
    def forward(self, front, sigma):
        return front + self.net(torch.cat([front, _sigma_map(sigma, front)], 1))


def build(mode, base=16, depth=3, sep=True):
    return {"learned": LearnedISP, "hybrid": HybridISP}[mode](base, depth, sep)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = build(ck["mode"], ck["base"], ck["depth"], ck["sep"]); m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck


def count_params(m): return sum(p.numel() for p in m.parameters())

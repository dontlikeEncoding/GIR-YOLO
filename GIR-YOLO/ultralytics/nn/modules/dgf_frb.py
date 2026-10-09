import torch
import torch.nn as nn
import torch.nn.functional as F


class FRBBranch(nn.Module):
    """Lightweight frequency/illumination robust feature recalibration."""
    def __init__(self, c1, strength=0.1):
        super().__init__()
        self.strength = float(strength)
        self.dw = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - blur
        enh = self.act(self.bn(self.pw(self.dw(high))))
        return x + self.strength * torch.tanh(self.alpha) * enh


class ISPCueBranch(nn.Module):
    """ISP-inspired illumination cue branch for feature modulation."""
    def __init__(self, c1, reduction=8):
        super().__init__()
        hidden = max(c1 // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        cue = self.net(x)
        return x * (1.0 + torch.tanh(self.beta) * (cue - 0.5))


class FRB_ISP_FixedGate(nn.Module):
    """FRB + ISP cue with a learnable scalar gate, no dynamic gate."""
    def __init__(self, c1, strength=0.1, gate_init=-1.0):
        super().__init__()
        self.frb = FRBBranch(c1, strength=strength)
        self.isp = ISPCueBranch(c1)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x):
        frb_feat = self.frb(x) - x
        isp_feat = self.isp(x)
        gate = torch.sigmoid(self.gate)
        return isp_feat + gate * frb_feat


class DGF_FRB(nn.Module):
    """ISP-guided dynamic gated frequency-recalibration block."""
    def __init__(self, c1, strength=0.1, gate_init=-1.0, reduction=8):
        super().__init__()
        hidden = max(c1 // reduction, 8)
        self.strength = float(strength)

        self.dw = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

        self.isp_pool = nn.AdaptiveAvgPool2d(1)
        self.isp_mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, c1, 1),
        )

        self.gate_bias = nn.Parameter(torch.full((1, c1, 1, 1), float(gate_init)))
        self.lambda_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - blur
        enh = self.act(self.bn(self.pw(self.dw(high))))

        pooled = self.isp_pool(x)
        cue = self.isp_mlp(pooled)
        gate = torch.sigmoid(self.gate_mlp(pooled * cue) + self.gate_bias)

        return x + self.strength * torch.tanh(self.lambda_scale) * gate * enh


class ResidualDGF(nn.Module):
    """Residual illumination-guided feature compensation block."""
    def __init__(self, c1, strength=1.0, gate_init=-2.0, reduction=4, alpha_init=0.1):
        super().__init__()
        hidden = max(c1 // reduction, 16)
        self.strength = float(strength)
        self.mode = str(mode)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.reduce = nn.Conv2d(c1, hidden, 1, bias=False)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.expand = nn.Conv2d(hidden, c1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.bn2 = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(c1 * 2, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, c1, 1),
        )
        self.gate_bias = nn.Parameter(torch.full((1, c1, 1, 1), float(gate_init)))

    def forward(self, x):
        if not hasattr(self, 'mode'):
            self.mode = 'full'
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low

        delta = self.reduce(high)
        delta = self.act(self.bn1(delta))
        delta = self.dw(delta)
        delta = self.act(delta)
        delta = self.bn2(self.expand(delta))

        mean = self.pool(x)
        contrast = self.pool(high.abs())
        gate = torch.sigmoid(self.gate(torch.cat([mean, contrast], dim=1)) + self.gate_bias)

        return x + self.strength * self.alpha * gate * delta


class InputResidualDGF(nn.Module):
    """Identity-preserving input residual enhancement for low-light images."""
    def __init__(self, c1=3, strength=0.1, gate_init=-5.0, reduction=4, alpha_init=0.0, mode='full'):
        super().__init__()
        hidden = max(int(c1 * reduction), 8)
        self.strength = float(strength)
        self.mode = str(mode)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.delta = nn.Sequential(
            nn.Conv2d(c1, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(c1 * 2, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, c1, 1),
        )
        self.gate_bias = nn.Parameter(torch.full((1, c1, 1, 1), float(gate_init)))

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        delta = self.delta(high)
        mean = self.pool(x)
        contrast = self.pool(high.abs())
        if not hasattr(self, 'mode'):
            self.mode = 'full'
        if self.mode == 'direct_h':
            return x + self.strength * self.alpha * high

        if self.mode == 'r_only':
            return x + self.strength * self.alpha * delta

        gate = torch.sigmoid(self.gate(torch.cat([mean, contrast], dim=1)) + self.gate_bias)
        return x + self.strength * self.alpha * gate * delta

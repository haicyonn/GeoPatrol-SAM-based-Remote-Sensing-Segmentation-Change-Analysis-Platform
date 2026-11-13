"""Small prompt-prior replacements used in the WHU resume experiment.

Haar prompting is inspired by SAMWave (BMVC 2025):
https://github.com/saurabhya/SAMWave
Soft spectral modulation is inspired by FE-SAM (TGRS 2026):
https://github.com/oucailab/FE-SAM
These reduced variants are not exact reproductions of either paper.
"""

import torch
from torch import nn
from torch.nn import functional as F


class HaarHighPrompt(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        filters = torch.tensor([
            [[1., 1.], [1., 1.]],
            [[-1., -1.], [1., 1.]],
            [[-1., 1.], [-1., 1.]],
            [[1., -1.], [-1., 1.]],
        ]).unsqueeze(1) / 2
        self.register_buffer("filters", filters.repeat(channels, 1, 1, 1))
        self.band_logits = nn.Parameter(torch.zeros(3))
        self.channels = channels

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels or h % 2 or w % 2:
            raise ValueError("Haar prior requires matching channels and even H,W")
        with torch.autocast(device_type=x.device.type, enabled=False):
            coeff = F.conv2d(x.float(), self.filters.float(), stride=2, groups=c)
            coeff = coeff.reshape(b, c, 4, h // 2, w // 2)
            high_weights = 3 * self.band_logits.float().softmax(0)
            weights = torch.cat((high_weights.new_zeros(1), high_weights))
            coeff = coeff * weights.reshape(1, 1, 4, 1, 1)
            high = F.conv_transpose2d(
                coeff.reshape(b, 4 * c, h // 2, w // 2),
                self.filters.float(), stride=2, groups=c,
            )
        return high.abs().to(x.dtype)

    def learned_filter_values(self):
        return {"direction_weights": (3 * self.band_logits.detach().softmax(0)).cpu().tolist()}


class SoftFrequencyPrompt(nn.Module):
    def __init__(self, temperature=0.02, cutoff_range=(0.08, 0.40)):
        super().__init__()
        self.cutoff_raw = nn.Parameter(torch.tensor(0.0))
        self.gain_raw = nn.Parameter(torch.tensor([-4.0, 0.0]))
        self.temperature = temperature
        self.cutoff_range = cutoff_range

    def forward(self, x):
        h, w = x.shape[-2:]
        with torch.autocast(device_type=x.device.type, enabled=False):
            fy = torch.fft.fftfreq(h, device=x.device).reshape(h, 1)
            fx = torch.fft.fftfreq(w, device=x.device).reshape(1, w)
            radius = (fy.square() + fx.square()).sqrt()
            lo, hi = self.cutoff_range
            cutoff = lo + (hi - lo) * self.cutoff_raw.sigmoid()
            low = ((cutoff - radius) / self.temperature).sigmoid()
            gains = 2 * self.gain_raw.sigmoid()
            spectrum = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
            output = torch.fft.ifft2(
                spectrum * (gains[0] * low + gains[1] * (1 - low)),
                dim=(-2, -1), norm="ortho",
            ).real
        return output.abs().to(x.dtype)

    def learned_filter_values(self):
        lo, hi = self.cutoff_range
        gains = 2 * self.gain_raw.detach().sigmoid()
        return {"cutoff_cycles_per_input_pixel":
                (lo + (hi - lo) * self.cutoff_raw.detach().sigmoid()).item(),
                "low_gain": gains[0].item(), "high_gain": gains[1].item()}


class StructureConditionedFrequencyPrompt(nn.Module):
    """Stage-wise body/boundary-conditioned frequency routing, proposed variant.

    Current SAM tokens are detached for the private structural predictor:
    auxiliary labels train this predictor, not the original SAM/Adapter stream.
    Zero-initialized output heads make the initial routed prior EXACTLY H.
    This combines established ideas; no claim of first-of-its-kind novelty.
    """

    def __init__(self, token_dim=1024, channels=32, stages=4):
        super().__init__()
        self.token_norm = nn.LayerNorm(token_dim)
        self.down = nn.Linear(token_dim, channels)
        self.spatial_norm = nn.GroupNorm(4, channels)
        self.local3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.local5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.mix = nn.Conv2d(channels, channels // 2, 1, bias=False)
        self.structure_head = nn.Conv2d(channels // 2, 2, 1)
        nn.init.zeros_(self.structure_head.weight)
        nn.init.zeros_(self.structure_head.bias)
        self.low_projection = nn.Conv2d(3, channels, 1, bias=False)
        self.route_scale = nn.Parameter(torch.full((stages, 2), 0.05))
        self.last_masks = {}

    def low_feature(self, image, size):
        # Average-pool within SAM patch support, a low-frequency appearance basis.
        return self.low_projection(F.adaptive_avg_pool2d(image, size))

    def forward(self, high, low, current_tokens, stage):
        z = self.down(self.token_norm(current_tokens.detach())).permute(0, 3, 1, 2)
        z = F.gelu(self.spatial_norm(z))
        z = 0.5 * (self.local3(z) + self.local5(z))
        logits = self.structure_head(F.gelu(self.mix(z)))
        body, edge = logits.float().sigmoid().split(1, dim=1)
        # Spatial centering prevents these maps from being only a global gain.
        body = body - body.mean((-2, -1), keepdim=True)
        edge = edge - edge.mean((-2, -1), keepdim=True)
        scales = self.route_scale[stage].float().tanh()
        routed = high.float() * (1 + scales[0] * edge) + scales[1] * body * low.float()
        self.last_masks[stage] = (body.detach(), edge.detach())
        return routed.to(high.dtype), logits

    def learned_filter_values(self):
        result = {"stage_route_scales": self.route_scale.detach().tanh().cpu().tolist()}
        result['spatial_gate_std'] = {
            str(stage): [body.std().item(), edge.std().item()]
            for stage, (body, edge) in self.last_masks.items()
        }
        return result

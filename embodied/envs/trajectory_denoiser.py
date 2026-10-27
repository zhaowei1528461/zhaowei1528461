"""Direct-trajectory adaptation of uploaded DiffScene temporal/context blocks.

Reference: diffscene-main/models/temporal.py and models/helpers.py. The uploaded
stride-one temporal hierarchy is retained; no nuScenes/CVAE/PyG dependencies.
"""
import math

import torch
from torch import nn


def _mlp(inp, out):
    return nn.Sequential(nn.Linear(inp, out), nn.Mish(), nn.Linear(out, out))


class ConditionEncoder(nn.Module):
    def __init__(self, history, condition_dim, mean, std):
        super().__init__()
        self.history = history
        self.register_buffer('state_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('state_std', torch.as_tensor(std, dtype=torch.float32))
        self.vehicle = _mlp(history*10, condition_dim)  # 9 derived fields + visibility.
        self.map_encoder = _mlp(6, condition_dim)
        self.missing_map = nn.Parameter(torch.zeros(condition_dim))
        self.scene_type = nn.Embedding(4, 8)
        self.combine = _mlp(4*condition_dim+8, condition_dim)

    def _vehicle(self, states, mask):
        mask = mask.bool()
        if states.shape[-2:] != (self.history, 8) or tuple(mask.shape) != tuple(states.shape[:-1]):
            raise ValueError('Historical state/mask shape mismatch')
        if not torch.isfinite(states[mask]).all():
            raise ValueError('Nonfinite active historical state')
        safe = torch.where(mask[..., None], states, torch.zeros_like(states))
        yaw = safe[..., 4:5]*self.state_std[4]+self.state_mean[4]
        features = torch.cat((safe[..., :4], torch.sin(yaw), torch.cos(yaw), safe[..., 5:]), dim=-1)
        features = torch.where(mask[..., None], features, torch.zeros_like(features))
        features = torch.cat((features, mask[..., None].to(features.dtype)), dim=-1)
        return self.vehicle(features.flatten(-2))

    def forward(self, condition):
        ego = self._vehicle(condition['ego_history'], condition['ego_mask'])
        target = self._vehicle(condition['target_history'], condition['target_mask'])
        values = condition['neighbor_history'].permute(0, 2, 1, 3)
        masks = condition['neighbor_mask'].permute(0, 2, 1).bool()
        if values.shape[1]:
            neighbors = self._vehicle(values, masks)
            visible = masks.any(dim=-1)
            neighbors = torch.where(visible[..., None], neighbors, torch.zeros_like(neighbors))
            neighbor = neighbors.sum(dim=1)/visible.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            neighbor = torch.zeros_like(ego)
        map_mask = condition['map_mask'].bool()
        segments = condition['map_segments']
        if not torch.isfinite(segments[map_mask]).all():
            raise ValueError('Nonfinite active map segment')
        safe_map = torch.where(map_mask[..., None], segments, torch.zeros_like(segments))
        encoded = self.map_encoder(safe_map)
        encoded = torch.where(map_mask[..., None], encoded, torch.zeros_like(encoded))
        map_feature = encoded.sum(dim=1)/map_mask.sum(dim=1, keepdim=True).clamp(min=1)
        map_feature = torch.where(map_mask.any(dim=1, keepdim=True), map_feature, self.missing_map[None])
        scene = condition['scene_type'].long()
        if torch.any((scene < 0)|(scene >= 4)):
            raise ValueError('Unknown scene-type token')
        return self.combine(torch.cat((ego, target, neighbor, map_feature, self.scene_type(scene)), dim=-1))


class SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError('Time embedding width must be even and >=4')
        self.register_buffer('frequency', torch.exp(torch.arange(dim//2)*(-math.log(10000)/(dim//2-1))))

    def forward(self, time):
        phase = time[:, None]*self.frequency[None]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


def _conv(inp, out):
    return nn.Sequential(nn.Conv1d(inp, out, 5, padding=2), nn.GroupNorm(math.gcd(8, out), out), nn.Mish())


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp, out, time_dim, condition_dim):
        super().__init__()
        self.first, self.second = _conv(inp, out), _conv(out, out)
        self.context = nn.Sequential(nn.Mish(), nn.Linear(time_dim+condition_dim, out))
        self.residual = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x, condition, time):
        h = self.first(x)+self.context(torch.cat((time, condition), dim=-1))[..., None]
        return self.second(h)+self.residual(x)


class ConditionalTrajectoryDenoiser(nn.Module):
    def __init__(self, config, normalizer):
        super().__init__()
        self.future = int(config['future'])
        dim, cond = int(config.get('base_dim', 32)), int(config.get('condition_dim', 64))
        multipliers = config.get('dim_mults', [1, 2, 4])
        if dim < 4 or dim % 2 or cond < 1 or not multipliers or any(m < 1 for m in multipliers):
            raise ValueError('Invalid temporal network dimensions')
        self.condition_encoder = ConditionEncoder(int(config['history']), cond, normalizer['mean'], normalizer['std'])
        self.time = nn.Sequential(SinusoidalTime(dim), nn.Linear(dim, dim*4), nn.Mish(), nn.Linear(dim*4, dim))
        widths = [2]+[dim*int(m) for m in multipliers]
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        for i, (inp, out) in enumerate(zip(widths[:-1], widths[1:])):
            self.downs.append(nn.ModuleList([ResidualTemporalBlock(inp, out, dim, cond),
                ResidualTemporalBlock(out, out, dim, cond),
                nn.Conv1d(out, out, 3, padding=1) if i < len(multipliers)-1 else nn.Identity()]))
        self.mid1, self.mid2 = (ResidualTemporalBlock(widths[-1], widths[-1], dim, cond) for _ in range(2))
        for inp, out in reversed(list(zip(widths[:-1], widths[1:]))[1:]):
            self.ups.append(nn.ModuleList([ResidualTemporalBlock(2*out, inp, dim, cond),
                ResidualTemporalBlock(inp, inp, dim, cond), nn.ConvTranspose1d(inp, inp, 3, stride=1, padding=1)]))
        self.final = nn.Sequential(_conv(widths[1], dim), nn.Conv1d(dim, 2, 1))

    def encode_condition(self, condition):
        return self.condition_encoder(condition)

    def predict_noise(self, noisy, time, context):
        if noisy.ndim != 3 or tuple(noisy.shape[1:]) != (self.future, 2) or time.shape != noisy.shape[:1]:
            raise ValueError('Expected noisy [B,F,2] and timestep [B]')
        x, t, skips = noisy.transpose(1, 2), self.time(time), []
        for first, second, conv in self.downs:
            x = second(first(x, context, t), context, t)
            skips.append(x)
            x = conv(x)
        x = self.mid2(self.mid1(x, context, t), context, t)
        for first, second, conv in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = conv(second(first(x, context, t), context, t))
        return self.final(x).transpose(1, 2)

    def forward(self, noisy, time, condition):
        return self.predict_noise(noisy, time, self.encode_condition(condition))

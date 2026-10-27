"""Zero-based DDPM noise prediction and actual DDIM skipping for direct x/y."""
import math

import torch
from torch import nn

try:
    from .trajectory_denoiser import ConditionalTrajectoryDenoiser
except ImportError:
    from trajectory_denoiser import ConditionalTrajectoryDenoiser


def _extract(values, time, shape):
    return values[time].reshape(len(time), *([1]*(len(shape)-1)))


class TrajectoryDiffusion(nn.Module):
    def __init__(self, denoiser, config):
        super().__init__()
        self.denoiser, self.config = denoiser, dict(config)
        self.num_steps, self.future = int(config.get('diffusion_steps', 100)), int(config['future'])
        if self.num_steps < 2:
            raise ValueError('At least two diffusion steps required')
        schedule = config.get('beta_schedule', 'cosine')
        if schedule == 'linear':
            betas = torch.linspace(config.get('beta_start', 1e-4), config.get('beta_end', .02), self.num_steps, dtype=torch.float64)
        elif schedule == 'cosine':
            time = torch.arange(self.num_steps+1, dtype=torch.float64)/self.num_steps
            abar = torch.cos((time+.008)/1.008*math.pi/2)**2
            abar = abar/abar[0]
            betas = (1-abar[1:]/abar[:-1]).clamp(max=.999)
        else:
            raise ValueError('Unknown beta schedule')
        if not torch.isfinite(betas).all() or torch.any((betas <= 0)|(betas >= 1)):
            raise ValueError('Invalid beta schedule')
        alphas = 1-betas
        bars = torch.cumprod(alphas, dim=0)
        prev = torch.cat((torch.ones(1, dtype=bars.dtype), bars[:-1]))
        for key, values in {'betas': betas, 'alphas': alphas, 'alpha_bars': bars,
            'posterior_variance': betas*(1-prev)/(1-bars),
            'posterior_c1': betas*torch.sqrt(prev)/(1-bars),
            'posterior_c2': torch.sqrt(alphas)*(1-prev)/(1-bars)}.items():
            self.register_buffer(key, values.float())

    def q_sample(self, start, time, noise=None):
        if time.dtype != torch.long or torch.any((time < 0)|(time >= self.num_steps)):
            raise ValueError('Diffusion timestep outside zero-based schedule')
        noise = torch.randn_like(start) if noise is None else noise
        bars = _extract(self.alpha_bars, time, start.shape)
        return bars.sqrt()*start+(1-bars).sqrt()*noise

    def q_posterior(self, start, noisy, time):
        return (_extract(self.posterior_c1, time, noisy.shape)*start+
                _extract(self.posterior_c2, time, noisy.shape)*noisy,
                _extract(self.posterior_variance, time, noisy.shape))

    def training_loss(self, condition, future_xy, future_mask=None, t=None, noise=None):
        if future_xy.ndim != 3 or tuple(future_xy.shape[1:]) != (self.future, 2):
            raise ValueError('Future labels must be [B,F,2]')
        mask = torch.ones_like(future_xy[..., 0], dtype=torch.bool) if future_mask is None else future_mask.bool()
        if tuple(mask.shape) != tuple(future_xy.shape[:-1]) or not mask.any():
            raise ValueError('A valid future mask is required')
        if not torch.isfinite(future_xy[mask]).all():
            raise ValueError('Nonfinite active future label')
        start = torch.where(mask[..., None], future_xy, torch.zeros_like(future_xy))
        t = torch.randint(self.num_steps, (len(start),), device=start.device) if t is None else t
        noise = torch.randn_like(start) if noise is None else noise
        prediction = self.denoiser(self.q_sample(start, t, noise), t, condition)
        squared = (prediction-noise)**2
        return torch.where(mask[..., None], squared, torch.zeros_like(squared)).sum()/(mask.sum()*2)

    @torch.no_grad()
    def sample(self, condition, num_samples=1, method='ddim', steps=None, seed=0, eta=0.):
        steps = self.num_steps if steps is None else int(steps)
        if num_samples < 1 or steps < 1 or steps > self.num_steps or eta < 0 or eta > 1:
            raise ValueError('Invalid sampling count/steps/eta')
        if method not in ('ddpm', 'ddim') or (method == 'ddpm' and steps != self.num_steps):
            raise ValueError('DDPM requires every diffusion step; use DDIM for skipping')
        training = self.training
        self.eval()
        try:
            context = self.denoiser.encode_condition(condition)
            batch = len(context)
            context = context.repeat_interleave(num_samples, dim=0)
            generator = torch.Generator(device=context.device).manual_seed(int(seed))
            shape = (batch*num_samples, self.future, 2)
            x = torch.randn(shape, device=context.device, dtype=context.dtype, generator=generator)
            times = (list(range(self.num_steps-1, -1, -1)) if method == 'ddpm' else
                     torch.linspace(self.num_steps-1, 0, steps).round().long().tolist())
            for i, current in enumerate(times):
                t = torch.full((len(x),), current, device=x.device, dtype=torch.long)
                epsilon = self.denoiser.predict_noise(x, t, context)
                bar = self.alpha_bars[current]
                start = (x-torch.sqrt(1-bar)*epsilon)/torch.sqrt(bar)
                if method == 'ddpm':
                    mean, variance = self.q_posterior(start, x, t)
                    x = mean
                    if current:
                        x = x+variance.sqrt()*torch.randn(shape, device=x.device, dtype=x.dtype, generator=generator)
                else:
                    previous = times[i+1] if i+1 < len(times) else -1
                    prev_bar = self.alpha_bars[previous] if previous >= 0 else torch.ones_like(bar)
                    sigma = eta*torch.sqrt(((1-prev_bar)/(1-bar)*(1-bar/prev_bar)).clamp(min=0))
                    x = prev_bar.sqrt()*start+torch.sqrt((1-prev_bar-sigma**2).clamp(min=0))*epsilon
                    if eta and previous >= 0:
                        x = x+sigma*torch.randn(shape, device=x.device, dtype=x.dtype, generator=generator)
            return x.reshape(batch, num_samples, self.future, 2)
        finally:
            self.train(training)


def make_model(config, normalizer):
    if normalizer.get('coordinate_frame') != 'ego-centric':
        raise ValueError('Direct trajectory diffusion requires an ego-centric normalizer')
    mean = torch.as_tensor(normalizer['mean'])
    std = torch.as_tensor(normalizer['std'])
    if mean.shape != (8,) or std.shape != (8,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid eight-field normalizer')
    return TrajectoryDiffusion(ConditionalTrajectoryDenoiser(config, normalizer), config)


def collate_samples(samples):
    import numpy as np
    condition = {key: torch.from_numpy(np.stack([s['condition'][key] for s in samples]))
                 for key in samples[0]['condition']}
    return {'condition': condition, 'future_xy': torch.from_numpy(np.stack([s['future_xy'] for s in samples])),
            'future_mask': torch.from_numpy(np.stack([s['future_mask'] for s in samples])),
            'meta': [s['meta'] for s in samples]}


def to_device(batch, device):
    return {'condition': {key: value.to(device) for key, value in batch['condition'].items()},
            'future_xy': batch['future_xy'].to(device), 'future_mask': batch['future_mask'].to(device),
            'meta': batch['meta']}

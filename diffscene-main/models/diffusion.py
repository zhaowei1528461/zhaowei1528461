import csv
import math

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Parameter
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from .helpers import extract, apply_conditioning

class VarianceSchedule(Module):
    def __init__(self, num_steps, mode="cosine", beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ("linear", "cosine")
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == "cosine":
            timesteps = torch.arange(num_steps+1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0) #Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)): # 1 to T
            log_alphas[i] += log_alphas[i-1]
        alphas_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alphas_bars[i - 1]) / (1 - alphas_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bars", alphas_bars)
        self.register_buffer("sigmas_flex", sigmas_flex) #DDPM
        self.register_buffer("sigmas_inflex", sigmas_inflex) #DDIM

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return  ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility <=1
        sigmas = self.sigmas_flex * flexibility + self.sigmas_inflex[t] * (1 - flexibility) #balance generation quality and speed
        return sigmas


class DiffusionTrainingLogger:
    def __init__(self,
                 root='./logs',
                 filename='diffusion_train'):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

        self.filename = filename
        if not self.filename.endswith('.csv'):
            self.filename += '.csv'

        self.filepath = os.path.join(self.root, self.filename)

        self.epoch_loss = []
        self.current_epoch = 0
        self.timestep = 0
        self.headers = [
            'epoch',
            'timestep',
            'loss'
        ]

        self._init_csv_file()

    def _init_csv_file(self):

        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def record_batch(self, loss):
        self.timestep += 1
        record = {
            'epoch': self.current_epoch,
            'timestep': self.timestep,
            'loss': loss
        }
        self.epoch_loss.append(record)

    def _save_to_csv(self):
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)

            for record in self.epoch_loss:
                writer.writerow(record)

        self.epoch_loss = []
        self.current_epoch += 1

class DiffusionLatent(Module):
    def __init__(self, net, var_sched: VarianceSchedule, observation_dim, action_dim, predict_epsilon=True):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.predict_epsilon = predict_epsilon
        self.train_log = DiffusionTrainingLogger()


    # ==================sample=====================================#

    def predict_start_from_noise(self, z_t, t, noise):
        '''
            if self.predict_epsilon, model output is noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            sqrt_recip_alphas_bar = torch.sqrt(1. / self.var_sched.alphas_bars)
            sqrt_recipm1_alphas_bar = torch.sqrt(1. / self.var_sched.alphas_bars - 1)
            return (
                extract(sqrt_recip_alphas_bar, t, z_t.shape) * z_t -
                extract(sqrt_recipm1_alphas_bar, t, z_t.shape) * noise
            )

        else:
            return noise

    def q_posterior(self, z_start, z_t, t):

        alphas_bar_prev = torch.cat([torch.ones(1, device=self.var_sched.alphas_bars.device), self.var_sched.alphas_bars[:-1]])
        posterior_mean_coef1 = self.var_sched.betas * torch.sqrt(alphas_bar_prev) / (1. - self.var_sched.alphas_bars)
        posterior_mean_coef1 = posterior_mean_coef1[1:]

        posterior_mean_coef2 = (1. - alphas_bar_prev) * torch.sqrt(self.var_sched.alphas) / (1. - self.var_sched.alphas_bars)
        posterior_mean_coef2 = posterior_mean_coef2[1:]

        posterior_mean = (
            extract(posterior_mean_coef1, t, z_t.shape) * z_start +
            extract(posterior_mean_coef2, t, z_t.shape) * z_t
        )

        posterior_variance = self.var_sched.betas * (1. - alphas_bar_prev) / (1. - self.var_sched.alphas_bars)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_log_variance_clipped = posterior_log_variance_clipped[1:]

        posterior_variance = extract(posterior_variance, t, z_t.shape)
        posterior_log_variance_clipped = extract(posterior_log_variance_clipped, t, z_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, z, t, cond, guide_sample_fn, guidance_scale, info):

        predicted_noise = self.net(z, cond, t) # predict noise without guidance

        if not z.requires_grad:
            z = z.detach().requires_grad_(True)

        if guide_sample_fn is not None and guidance_scale > 0:
            loss, guidance_grad = guide_sample_fn.calculate_loss(z, **info)
            predicted_noise = predicted_noise - guidance_scale * guidance_grad


        z_start_recon = self.predict_start_from_noise(z, t=t, noise=predicted_noise) # predict z_0

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(z_start=z_start_recon, z_t=z, t=t)

        return model_mean, posterior_variance, posterior_log_variance # compute posterior mean and posterior variance

    @torch.no_grad()
    def p_sample_step(self, z, cond, t, guide_sample_fn, guidance_scale, info):
        model_mean, model_variance, model_log_variance = self.p_mean_variance(z=z, t=t, cond=cond, guide_sample_fn=guide_sample_fn, guidance_scale=guidance_scale, info=info)
        model_std = torch.exp(0.5 * model_log_variance)

        if t[0].detach() > 0:
            noise = torch.randn_like(z)
            z_prev = model_mean + model_std * noise # z_k-1 = u_t + sigma * epsilon
        else:
            z_prev = model_mean

        return z_prev # compute z_t-1


    @torch.no_grad()
    def p_sample_loop(self,
                      shape,
                      cond = None,
                      guide_sample_fn = None,
                      guidance_scale = None,
                      guide_start_step = 0,
                      sample_size = 5,
                      info = None):
        device = self.var_sched.betas.device

        batch_size = shape[0]
        new_shape = [shape[0], sample_size, shape[1]]
        z = torch.randn(new_shape, device=device) # add noise
        # z = apply_conditioning(z, cond, self.action_dim) # z_t
        for i in range(self.var_sched.num_steps-1, -1, -1):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)


            current_guidance_scale = guidance_scale if (i >= guide_start_step and
                                                        guide_sample_fn is not None) else 0.0 # start guiding step
            z = self.p_sample_step(z, cond, t, guide_sample_fn, current_guidance_scale, info)
        return z

    #==================train=====================================#

    def q_sample(self, z_start, t, noise=None): #add_noise

        if noise is None:
            noise = torch.randn_like(z_start)

        sqrt_alphas_bar = torch.sqrt(self.var_sched.alphas_bars)
        sqrt_one_minus_alphas_bar = torch.sqrt(1-self.var_sched.alphas_bars)

        # device = sqrt_alphas_bar.device
        # t = t.to(device)
        # z_start = z_start.to(device)
        # noise = noise.to(device)
        z_t = (
                extract(sqrt_alphas_bar, t, z_start.shape) * z_start +
                extract(sqrt_one_minus_alphas_bar, t, z_start.shape) * noise)

        return z_t


    def p_losses(self, z_start, cond, t, sample_size=20): #caculate training loss
        # reshape z_0 to (num_adv, latent_dim) -> (num_adv, sample_size, latent_dim)
        z_start = z_start.unsqueeze(1).repeat(1, sample_size, 1)

        noise = torch.randn_like(z_start, device=z_start.device)

        z_noisy = self.q_sample(z_start=z_start, t=t, noise=noise)
        # z_noisy = apply_conditioning(z_noisy, cond, self.action_dim)

        # cond = cond.to(z_noisy.device)
        # t = t.to(z_noisy.device)
        # print("device：",z_noisy.device)
        z_recon = self.net(z_noisy, cond, t)
        # z_recon = apply_conditioning(z_recon, cond, self.action_dim) #vanilla guidance

        noise = noise.to(z_recon)

        if self.predict_epsilon:
            loss = F.mse_loss(z_recon, noise, reduction='mean') # predict noise
        else:
            loss = F.mse_loss(z_recon, z_start, reduction='mean') # predict initial data
        return loss

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.var_sched.num_steps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self,
                cond,
                horizon=None,
                guide_sample_fn=None,
                guidance_scale=None,
                guide_start_step=0
                ):
        device = self.var_sched.betas.device
        batch_size = len(cond[0])
        horizon = horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape,
                      cond = cond,
                      guide_sample_fn = guide_sample_fn,
                      guidance_scale = guidance_scale,
                      guide_start_step = guide_start_step)




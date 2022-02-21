#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""docstring summary

"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from DCT import dct, idct


# TODO: channel wise operation
# TODO: repr
# TODO: median filter - baseline / Hz
# TODO: DynamicTimeWarp

def get_random_integer(high):
    return torch.randint(high, (1,)).item()


def random_crop(x, size):
    # x.shape : (n_channel, n_time_stamps)
    assert x.shape[1] >= size
    st_pt = get_random_integer(x.shape[1] - size + 1)
    return x[:, st_pt:st_pt + size]


class DCTDelete(nn.Module):
    def __init__(self, slices: list, eps=1e-7):
        super(DCTDelete, self).__init__()
        self.slices = slices
        self.eps = eps

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        original_std = x.std()
        x = dct(x[None, ...])
        for el in self.slices:
            x[..., el] = 0
        x = idct(x)[0]
        x = x / (x.std() + self.eps)
        return x * original_std

    def __repr__(self):
        return super(DCTDelete, self).__repr__()


class MedianFilter1D(nn.Module):
    def __init__(self, kernel_size):
        super(MedianFilter1D, self).__init__()
        self.Padding = nn.ReplicationPad1d(padding=(kernel_size - 1) // 2)
        self.Unfold = nn.Unfold(kernel_size=(1, kernel_size))

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        return torch.median(self.Unfold(self.Padding(x)[:, None, None, ...]), 1)[0]

    def __repr__(self):
        return super(MedianFilter1D, self).__repr__()


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.mf1 = MedianFilter1D(101)
        self.mf2 = MedianFilter1D(301)

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        return self.mf2(self.mf1(x))

    def __repr__(self):
        return super(Baseline, self).__repr__()


class GaussianFilter1D(nn.Module):
    def __init__(self, sigma, truncate=4, mode='replicate'):
        # mode:  'zeros', 'reflect', 'replicate' or 'circular'
        super(GaussianFilter1D, self).__init__()
        self.P = Normal(torch.tensor([0]), torch.tensor([sigma]))
        kernel_length = int(truncate * sigma + 0.5)
        weight = self.P.log_prob(torch.arange(-kernel_length, kernel_length + 1)).exp()
        self.conv = nn.Conv1d(1, 1, kernel_size=len(weight), bias=False, padding=kernel_length, padding_mode=mode,
                              dtype=torch.float)
        self.conv.weight = nn.Parameter(
            (weight[None, None, :] / torch.sum(weight)).float()
        )
        self.conv.requires_grad_(False)

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        return self.conv(x[:, None, :])[:, 0, :]


class Resize(nn.Module):
    def __init__(self, size, interpolation='linear'):
        super(Resize, self).__init__()
        self.size = size
        self.mode = interpolation

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        return F.interpolate(
            x[None, ...], size=self.size, mode=self.mode, align_corners=False
            # x[None, ...], size = self.size, mode = self.mode, align_corners = False, recompute_scale_factor = None
        )[0]

    def __repr__(self):
        return super(Resize, self).__repr__()


class RandomCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        return random_crop(x, self.size)

    def __repr__(self):
        return super(RandomCrop, self).__repr__()


class RandomResizedCrop(nn.Module):
    def __init__(self, size, scale=(0.5, 1.0), interpolation='linear'):
        super(RandomResizedCrop, self).__init__()
        self.size = size
        self.scale = scale
        assert len(scale) == 2 and (0 < scale[0] < scale[1] <= 1)
        self.resizer = Resize(size, interpolation=interpolation)

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        size = int(x.shape[1] * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        x = random_crop(x, size)
        return self.resizer(x)

    def __repr__(self):
        return super(RandomResizedCrop, self).__repr__()


class TimeOut(nn.Module):
    def __init__(self, n_timeout, max_timestamp, prob=0.5):
        super(TimeOut, self).__init__()
        self.n_timeout = n_timeout
        self.prob = prob
        self.max_timestamp = max_timestamp

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        x = x.clone()
        for _ in range(self.n_timeout):
            if torch.rand(1) > self.prob:
                continue
            st = get_random_integer(x.shape[1] - self.max_timestamp + 1)
            du = get_random_integer(self.max_timestamp) + st
            x[:, st:du] = 0
        return x

    def __repr__(self):
        return super(TimeOut, self).__repr__()


class DownSampleReconstruct(nn.Module):
    def __init__(self, downsample_size, original_size, interpolation='linear'):
        super(DownSampleReconstruct, self).__init__()
        self.size = downsample_size
        self.original_size = original_size
        self.resizer1 = Resize(downsample_size, interpolation=interpolation)
        self.resizer2 = Resize(original_size, interpolation=interpolation)

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        return self.resizer2(self.resizer1(x))

    def __repr__(self):
        return super(DownSampleReconstruct, self).__repr__()


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def __repr__(self):
        return super(GaussianNoise, self).__repr__()

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        return x + (self.sigma * torch.randn_like(x))


class PowerlineNoise(nn.Module):
    def __init__(self, wave_freq, noise_freq=(80, 100), scale=0.1, n_noise=3):
        super(PowerlineNoise, self).__init__()
        self.wave_freq = wave_freq
        self.noise_freq = noise_freq
        self.scale = scale
        self.n_noise = n_noise
        self.freq_ratio = [wave_freq / el for el in noise_freq]

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        noise_freq = torch.empty(1).uniform_(self.freq_ratio[1], self.freq_ratio[0]).item()
        for _ in range(self.n_noise):
            noise = torch.cos(
                torch.linspace(0, (x.shape[1] + 1) * 2 * torch.pi / noise_freq, x.shape[1])
            ).to(x.device)
            x = x + noise * self.scale * torch.rand(1, ).item()
        return x

    def __repr__(self):
        return super(PowerlineNoise, self).__repr__()


class BaselineWander(nn.Module):
    def __init__(self, random=True):
        super(BaselineWander, self).__init__()
        self.baseline = Baseline()
        self.random = random

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        if self.random:
            return x - (torch.rand(1, ).item() * self.baseline(x))
        else:
            return x - self.baseline(x)


class RandomTF(nn.Module):
    def __init__(self, transform, p=0.5):
        super(RandomTF, self).__init__()
        self.transform = transform
        self.p = p

    def forward(self, x):
        # x.shape : (n_channel, n_time_stamps)
        if torch.rand(1, ).item() < self.p:
            return self.transform(x)
        else:
            return x

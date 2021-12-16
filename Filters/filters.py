#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""docstring summary

"""

import torch
from torch import nn
from torch.distributions.normal import Normal


class GaussianFilter1D(nn.Module):
    def __init__(self, sigma, length, truncate=4, mode='replicate'):
        """
        :param sigma:
        :param length:
        :param truncate:
        :param mode: 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        """
        super(GaussianFilter1D, self).__init__()
        self.P = Normal(torch.tensor([0]), torch.tensor([sigma]))
        assert length > 0
        kernel_length = int(truncate * sigma + 0.5)
        weight = self.P.log_prob(torch.arange(-kernel_length, kernel_length + 1)).exp()
        self.conv = nn.Conv1d(1, 1, kernel_size=len(weight), bias=False, padding=kernel_length, padding_mode=mode)
        self.conv.weight = nn.Parameter(
            weight[None, None, :] / torch.sum(weight)
        )
        self.conv.requires_grad_(False)

    def forward(self, x):
        x = self.conv(
            x.view(
                x.shape[0] * x.shape[1], 1, -1
            )
        ).view(*x.shape[:2], -1)
        return x


class MedianFilter1D(nn.Module):
    def __init__(self, kernel_size: int, padding_mode='replicate', chunk_size: int = None, middle_device='cpu',
                 output_device='cpu'):
        super(MedianFilter1D, self).__init__()

        assert kernel_size % 2 and kernel_size > 0
        self.kernel_size = kernel_size
        self.Unfold = nn.Unfold(kernel_size=(1, kernel_size))
        if chunk_size is not None:
            assert chunk_size > 0
        self.chunk_size = chunk_size
        self.middle_device = middle_device
        self.output_device = output_device
        if padding_mode == 'replicate':
            self.Padding = nn.ReplicationPad1d(padding=(kernel_size - 1) // 2)
        elif padding_mode == 'reflect':
            self.Padding = nn.ReflectionPad1d(padding=(kernel_size - 1) // 2)
        elif padding_mode == 'zeros':
            self.Padding = nn.ConstantPad1d(padding=(kernel_size - 1) // 2, value=0.0)
        else:
            raise NotImplementedError()

    def forward(self, x):
        if self.chunk_size is None:
            return torch.median(self.Unfold(
                self.Padding(
                    x.to(self.middle_device)
                )[:, None, ...]
            ).view(len(x), self.kernel_size, x.shape[1], x.shape[2]), 1)[0].to(self.output_device)
        else:
            tmp = []
            i = 0
            for _ in range(len(x) // self.chunk_size):
                tmp.append(
                    torch.median(self.Unfold(
                        self.Padding(
                            x[i * self.chunk_size:(i + 1) * self.chunk_size].to(self.middle_device)
                        )[:, None, ...]
                    ).view(-1, self.kernel_size, x.shape[1], x.shape[2]), 1)[0].to(self.output_device)
                )
                i += 1
            tmp.append(
                torch.median(self.Unfold(
                    self.Padding(
                        x[i * self.chunk_size:len(x)].to(self.middle_device)
                    )[:, None, ...]
                ).view(-1, self.kernel_size, x.shape[1], x.shape[2]), 1)[0].to(self.output_device)
            )
            return torch.cat(tmp)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.misc import electrocardiogram

    ECG_input = torch.as_tensor(np.asarray([[
        electrocardiogram()[2000:4000],
        electrocardiogram()[6000:8000],
        electrocardiogram()[8000:10000]
    ], [
        electrocardiogram()[12000:14000],
        electrocardiogram()[16000:18000],
        electrocardiogram()[18000:20000]
    ]])).float()  # 360 Hz
    print('ECG input shape :', ECG_input.shape)
    MF = MedianFilter1D(301, chunk_size=1)
    GF = GaussianFilter1D(4, 2000)
    MF_output = MF(ECG_input)
    GF_output = GF(ECG_input)

    fig, axs = plt.subplots(figsize=(10, 2))
    axs.plot(ECG_input[0, 0], label='Original')
    axs.plot(GF_output[0, 0], label='Gaussian Filter')
    axs.plot(MF_output[0, 0], label='Median Filter')
    plt.tight_layout()
    plt.legend()
    plt.show()

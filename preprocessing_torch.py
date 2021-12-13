from torch.distributions.normal import Normal
import torch
from torch import nn


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
    def __init__(self, kernel_size: int, padding_mode='replicate'):
        super(MedianFilter1D, self).__init__()

        assert kernel_size % 2 and kernel_size > 0
        self.kernel_size = kernel_size
        self.Unfold = nn.Unfold(kernel_size=(1, kernel_size))
        if padding_mode == 'replicate':
            self.Padding = nn.ReplicationPad1d(padding=(kernel_size - 1) // 2)
        elif padding_mode == 'reflect':
            self.Padding = nn.ReflectionPad1d(padding=(kernel_size - 1) // 2)
        elif padding_mode == 'zeros':
            self.Padding = nn.ConstantPad1d(padding=(kernel_size - 1) // 2, value=0.0)
        else:
            raise NotImplementedError()

    def forward(self, x):
        unfolded = self.Unfold(
            self.Padding(
                x
            )[:, None, ...]
        ).view(len(x), self.kernel_size, x.shape[1], x.shape[2])
        median_unfolded = torch.median(unfolded, 1)[0]
        return median_unfolded


class EcgPreprocess(nn.Module):
    '''
    500 hz ECG
    reference :
    Advances in Cardiac Signal Processing
    Book by Rajendra Acharya U
    '''

    def __init__(self, Hz=500):
        super(EcgPreprocess, self).__init__()

        self.param1 = round((Hz / 5 - 1) / 2) * 2 + 1
        self.param2 = round(((Hz * 3 / 5) - 1) / 2) * 2 + 1
        self.MF1 = MedianFilter1D(self.param1)
        self.MF2 = MedianFilter1D(self.param2)

    def forward(self, x):
        y = self.MF2(self.MF1(x))
        return x - y


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.misc import electrocardiogram

    ECG_input = np.asarray([[
        electrocardiogram()[2000:4000],
        electrocardiogram()[6000:8000],
        electrocardiogram()[8000:10000]
    ], [
        electrocardiogram()[12000:14000],
        electrocardiogram()[16000:18000],
        electrocardiogram()[18000:20000]
    ]])  # 360 Hz
    print('ECG input shape :', ECG_input.shape)

    EP = EcgPreprocess(360)
    EP_output = EP(torch.as_tensor(ECG_input)).detach()

    GF = GaussianFilter1D(2, 2000)
    GF_output = GF(EP_output.float())

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(ECG_input[0, 0])
    axs[0].set_title('original')
    axs[1].plot(EP_output[0, 0])
    axs[1].set_title('baseline wander remover')
    axs[2].plot(GF_output[0, 0])
    axs[2].set_title('denoise')
    plt.tight_layout()
    plt.show()

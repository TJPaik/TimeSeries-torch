from torch.distributions.normal import Normal
import torch
from torch import nn


class GaussianFilter1D(nn.Module):
    def __init__(self, sigma, fixed_length: int = -1):
        super(GaussianFilter1D, self).__init__()
        self.P = Normal(torch.tensor([0]), torch.tensor([sigma]))
        if fixed_length != -1:
            assert fixed_length > 0
            weight = self.P.log_prob(torch.arange(-fixed_length, fixed_length)).exp()
            self.conv = nn.Conv1d(1, 1, kernel_size=2 * fixed_length, bias=False)
            self.conv.weight = nn.Parameter(
                weight[None, None, :]
            )
            self.conv.requires_grad_(False)
        else:
            raise NotImplementedError()

    def forward(self, x):
        x = torch.cat(
            (
                torch.flip(x, (2,)),
                x,
                torch.flip(x[:, :, 1:], (2,))
            ),
            dim=2
        )
        x_shape = x.shape
        x = self.conv(
            x.view(
                x_shape[0] * x_shape[1], 1, -1
            )
        ).view(*x_shape[:2], -1)
        return x


class MedianFilter1D(nn.Module):
    def __init__(self, kernel_size: int):
        super(MedianFilter1D, self).__init__()
        assert kernel_size % 2 and kernel_size > 0
        self.kernel_size = kernel_size
        self.Unfold = nn.Unfold(kernel_size=(1, kernel_size))
        # self.Padding = nn.ReflectionPad1d(padding=(kernel_size - 1) // 2)
        # self.Padding = nn.ReplicationPad1d(padding=(kernel_size - 1) // 2)
        self.Padding = nn.ConstantPad1d(padding=(kernel_size - 1) // 2, value=0.0)

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
    reference :
    Advances in Cardiac Signal Processing
    Book by Rajendra Acharya U
    '''

    def __init__(self):
        super(EcgPreprocess, self).__init__()

        self.MF1 = MedianFilter1D(101)
        self.MF2 = MedianFilter1D(301)

    def forward(self, x):
        y = self.MF2(self.MF1(x))
        return x - y


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.signal import medfilt
    from scipy.ndimage import gaussian_filter1d
    from scipy.misc import electrocardiogram

    dummy_input = torch.rand(3, 6, 21)
    SIGMA = 2
    GF = GaussianFilter1D(sigma=SIGMA, fixed_length=21)
    output = GF(dummy_input).detach()
    original_output = gaussian_filter1d(dummy_input, sigma=SIGMA, axis=2)
    print('Gaussian Filter maximal error:', torch.max(torch.abs(torch.as_tensor(original_output) - output)).item())

    plt.plot(output[1, 4].detach(), label='torch version')
    plt.plot(original_output[1, 4], label='original')
    plt.title('Gaussian filter test')
    plt.legend()
    plt.show()

    dummy_input = torch.rand(3, 6, 28)
    kernel_size = 5
    MF = MedianFilter1D(kernel_size)
    output = MF(dummy_input).detach()
    original_output = np.asarray([[medfilt(dummy_input[i, j], kernel_size) for j in range(6)] for i in range(3)])
    print('Median Filter maximal error:', torch.max(torch.abs(torch.as_tensor(original_output) - output)).item())

    plt.plot(output[1, 4].detach(), label='torch version')
    plt.plot(original_output[1, 4], label='original')
    plt.title('Median Filter test')
    plt.legend()
    plt.show()

    x = np.asarray([[
        electrocardiogram()[2000:4000],
        electrocardiogram()[6000:8000],
        electrocardiogram()[8000:10000]
    ], [
        electrocardiogram()[12000:14000],
        electrocardiogram()[16000:18000],
        electrocardiogram()[18000:20000]
    ]])
    EP = EcgPreprocess()
    x_preprocessed = EP(torch.as_tensor(x))

    plt.plot(x[0, 0], label='original')
    plt.plot(x_preprocessed[0, 0], label='debased')
    plt.legend()
    plt.show()

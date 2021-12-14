#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""docstring summary

"""

import torch
from torch import nn

from Filters import MedianFilter1D, GaussianFilter1D


class EcgPreprocess(nn.Module):
    '''
    reference :
    Advances in Cardiac Signal Processing
    Book by Rajendra Acharya U
    '''

    def __init__(self, hz=500):
        super(EcgPreprocess, self).__init__()

        self.param1 = round((hz / 5 - 1) / 2) * 2 + 1
        self.param2 = round(((hz * 3 / 5) - 1) / 2) * 2 + 1
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

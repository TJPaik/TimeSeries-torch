#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""docstring summary

"""

import torch
from torch.fft import rfft, irfft


def dct(x, type=2):
    if type == 1:
        x = rfft(
            torch.cat([
                x,
                torch.flip(x, (2,))[..., 1:-1]
            ],
                dim=2
            )
        )
        return x.real
    elif type == 2:
        tmp = torch.zeros(*x.shape[:2], 4 * x.shape[2])
        tmp[..., 1::2][..., :x.shape[2]] = x
        tmp[..., x.shape[2] * 2 + 1:] = torch.flip(tmp[..., 1:x.shape[2] * 2], (2,))
        return rfft(tmp)[..., :x.shape[2]].real
    else:
        raise NotImplementedError()


def idct(x, type=2):
    if type == 1:
        return dct(x, type=type) / (2 * (x.shape[2] - 1))
    elif type == 2:
        w_length = x.shape[2]
        x = torch.cat((
            x,
            torch.zeros(*x.shape[:2], 1),
            -torch.flip(x, (2,))
        ), 2)
        return irfft(x)[..., 1::2][..., :w_length]
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    dummy_waves = torch.rand(3, 5, 1024)
    recon1 = idct(dct(dummy_waves, type=1), type=1)
    recon2 = idct(dct(dummy_waves, type=2), type=2)
    max_error = max(
        torch.max(torch.abs(recon1 - dummy_waves)),
        torch.max(torch.abs(recon2 - dummy_waves))
    )
    assert max_error < 1e-5
    print('error :', max_error)

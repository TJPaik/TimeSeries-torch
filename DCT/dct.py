from torch.fft import rfft
import torch


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
        raise NotImplementedError()
    else:
        raise NotImplementedError()

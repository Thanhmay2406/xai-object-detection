import torch.nn.functional as F


def get_padded_tensor(tensor, multiple):
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h))

import torch.nn.functional as F


def softmax_loss(logits, labels):
    return F.cross_entropy(logits, labels.long(), reduction="none")


def smooth_l1_loss(pred, target, beta):
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="none").sum(dim=1)

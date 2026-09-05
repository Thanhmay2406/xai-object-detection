"""Reusable fine-tuning primitives shared across pruning methods."""

from .finetune import train_one_epoch

__all__ = ["train_one_epoch"]

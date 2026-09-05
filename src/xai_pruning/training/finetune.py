"""Low-level fine-tuning functions; no experiment-specific Trainer abstraction."""

from __future__ import annotations

from collections import defaultdict

import torch


def move_batch_to_device(batch, device: str | torch.device):
    """Move images and tensor target fields to one device."""

    images, targets = batch
    images = [image.to(device, non_blocking=True) for image in images]
    targets = [
        {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]
    return images, targets


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device: str | torch.device,
    *,
    scaler=None,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one detector-training epoch using the same loss sum for every method."""

    resolved_device = torch.device(device)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and resolved_device.type == "cuda")
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    model.train()
    running = defaultdict(float)
    steps = 0

    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, targets = move_batch_to_device(batch, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            losses = model(images, targets)
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Loss became non-finite: {total_loss.item()}")
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["loss_total"] += float(total_loss.detach().cpu())
        for key, value in losses.items():
            running[key] += float(value.detach().cpu())
        steps += 1

    return {key: value / max(steps, 1) for key, value in running.items()}

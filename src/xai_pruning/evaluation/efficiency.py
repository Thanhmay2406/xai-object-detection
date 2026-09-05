"""Parameter, Conv/Linear MAC, peak-memory, and latency measurements."""

from __future__ import annotations

import gc
import time

import numpy as np
import torch
from torch import nn

from xai_pruning.pruning.structural import count_parameters
from xai_pruning.utils.device import synchronize


def estimate_conv_linear_macs(model, image_cpu, device: str | torch.device) -> int:
    """Count only Conv2d and Linear MACs for one detector forward pass.

    This intentionally is not described as exact total FLOPs: preprocessing,
    pooling, activations, NMS, and other operations are outside this counter.
    """

    total = {"macs": 0}
    handles = []

    def conv_hook(module, _inputs, output):
        if torch.is_tensor(output):
            batch, channels, height, width = (int(value) for value in output.shape)
            kernel_height, kernel_width = module.kernel_size
            input_channels_per_group = module.in_channels // module.groups
            total["macs"] += (
                batch
                * channels
                * height
                * width
                * input_channels_per_group
                * kernel_height
                * kernel_width
            )

    def linear_hook(module, _inputs, output):
        if torch.is_tensor(output):
            total["macs"] += int(output.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    try:
        with torch.inference_mode():
            model([image_cpu.to(resolved_device)])
        synchronize(resolved_device)
    finally:
        for handle in handles:
            handle.remove()
    return int(total["macs"])


def measure_peak_memory(model, images_cpu, device: str | torch.device) -> float | None:
    """Measure peak allocated CUDA memory; return None for CPU."""

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        return None
    model.to(resolved_device).eval()
    torch.cuda.reset_peak_memory_stats(resolved_device)
    with torch.inference_mode():
        for image in images_cpu:
            model([image.to(resolved_device)])
    synchronize(resolved_device)
    return float(torch.cuda.max_memory_allocated(resolved_device) / (1024**2))


def benchmark_latency(
    model,
    images_cpu,
    device: str | torch.device,
    *,
    model_name: str = "model",
    warmup: int = 10,
    repeats: int = 50,
    move_model_to_cpu_after: bool = False,
) -> dict:
    """Measure synchronized batch-one latency using Pipeline 04's protocol."""

    if not images_cpu:
        raise ValueError("At least one benchmark image is required")
    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    images_device = [image.to(resolved_device) for image in images_cpu]
    with torch.inference_mode():
        for index in range(warmup):
            model([images_device[index % len(images_device)]])
        synchronize(resolved_device)
        if resolved_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved_device)
        latencies_ms = []
        for index in range(repeats):
            image = images_device[index % len(images_device)]
            synchronize(resolved_device)
            started = time.perf_counter()
            model([image])
            synchronize(resolved_device)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
        peak_mb = (
            torch.cuda.max_memory_allocated(resolved_device) / (1024**2)
            if resolved_device.type == "cuda"
            else None
        )
    values = np.asarray(latencies_ms, dtype=np.float64)
    result = {
        "model": model_name,
        "latency_mean_ms": float(values.mean()),
        "latency_std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p90_ms": float(np.percentile(values, 90)),
        "fps_from_mean_latency": float(1000.0 / values.mean()),
        "peak_gpu_memory_mb": float(peak_mb) if peak_mb is not None else None,
        "latency_warmup": warmup,
        "latency_repeats": repeats,
        "benchmark_num_images": len(images_cpu),
    }
    if move_model_to_cpu_after:
        model.to("cpu")
        del images_device
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    return result

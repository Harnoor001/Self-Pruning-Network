"""Evaluation and efficiency measurement.

Definitions (kept deliberately explicit):

* **deployable parameters** — trainable tensors needed at inference: weights, biases
  and BatchNorm affine parameters. Gate scores and masks are training-only and excluded.
* **deployable size** — bytes of ``torch.save`` of the state dict restricted to
  deployable tensors plus BN running statistics (serialised in memory).
* **MACs** — multiply-accumulates per image of every ``Linear``/``Conv2d`` (and gated
  equivalents) in the *instantiated* model, counted with forward hooks. BN, activations
  and pooling are excluded. This is an architectural count, not a latency measurement.
* **latency** — measured wall-clock time; reported separately per device.
"""
from __future__ import annotations

import io
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn

from self_pruning_network.gates import PrunableConv2d, PrunableLinear

TRAINING_ONLY_KEYS = ("gate_scores", "mask", "hard_pruned")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    """Mean CE loss (no label smoothing) and accuracy."""
    was_training = model.training
    model.eval()
    loss_sum = correct = total = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(inputs).float()
        loss_sum += float(criterion(logits, targets))
        correct += float((logits.argmax(1) == targets).sum())
        total += targets.shape[0]
    model.train(was_training)
    return {"loss": loss_sum / max(1, total), "accuracy": correct / max(1, total), "n": int(total)}


def deployable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if not any(k in n for k in TRAINING_ONLY_KEYS))


def deployable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    seen: set[int] = set()
    state = {}
    for key, value in model.state_dict().items():
        if any(k in key for k in TRAINING_ONLY_KEYS):
            continue
        if value.data_ptr() in seen and value.numel():  # aliased tensors (legacy MLP duplicates)
            continue
        seen.add(value.data_ptr())
        state[key] = value.detach().cpu()
    return state


def deployable_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(deployable_state_dict(model), buffer)
    return buffer.getbuffer().nbytes


@torch.no_grad()
def count_macs(model: nn.Module, input_shape: tuple[int, ...] = (1, 3, 32, 32)) -> int:
    total = 0
    handles = []

    def linear_hook(module, inputs, output):
        nonlocal total
        total += module.in_features * module.out_features * (output.numel() // output.shape[-1] // output.shape[0])

    def conv_hook(module, inputs, output):
        nonlocal total
        kh, kw = module.kernel_size
        total += (output.shape[1] * output.shape[2] * output.shape[3]) * (module.in_channels // module.groups) * kh * kw

    for module in model.modules():
        if isinstance(module, (nn.Linear, PrunableLinear)):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.Conv2d, PrunableConv2d)):
            handles.append(module.register_forward_hook(conv_hook))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        model(torch.zeros(*input_shape, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return int(total // input_shape[0])


@torch.no_grad()
def measure_latency(model: nn.Module, batch_size: int, device: torch.device, warmup: int = 20,
                    iterations: int = 100, input_shape: tuple[int, ...] = (3, 32, 32)) -> dict[str, float]:
    """Median / IQR / mean latency in milliseconds. GPU timing uses CUDA events + synchronisation."""
    model.eval().to(device)
    inputs = torch.randn(batch_size, *input_shape, generator=torch.Generator().manual_seed(0)).to(device)
    for _ in range(warmup):
        model(inputs)
    times: list[float] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        for _ in range(iterations):
            t0 = time.perf_counter()
            model(inputs)
            times.append((time.perf_counter() - t0) * 1000.0)
    q = statistics.quantiles(times, n=4) if len(times) >= 2 else [times[0]] * 3
    return {"median_ms": statistics.median(times), "p25_ms": q[0], "p75_ms": q[2], "mean_ms": statistics.fmean(times),
            "iterations": iterations, "batch_size": batch_size}


@torch.no_grad()
def peak_gpu_memory_bytes(model: nn.Module, batch_size: int, device: torch.device,
                          input_shape: tuple[int, ...] = (3, 32, 32)) -> int | None:
    if device.type != "cuda":
        return None
    model.eval().to(device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    model(torch.randn(batch_size, *input_shape, device=device))
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated(device))


def environment_metadata() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    meta: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "cpu_threads": torch.get_num_threads(),
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        meta.update({"gpu_name": props.name, "gpu_total_memory_bytes": int(props.total_memory),
                     "gpu_free_memory_bytes": int(torch.cuda.mem_get_info()[0]), "gpu_count": torch.cuda.device_count()})
    else:
        meta["gpu_note"] = "No CUDA device available: GPU latency/memory were NOT measured."
    return meta

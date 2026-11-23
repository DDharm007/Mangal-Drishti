"""Hardware and inference latency/memory benchmarking for MANGAL DRISHTI models."""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil
import torch

from src.data.config_utils import get_project_root
from src.models.deeplabv3plus import DeepLabV3PlusResNet50
from src.models.unet import UNetResNet34
from src.training.checkpoint import load_checkpoint

LOGGER = logging.getLogger("mangal_drishti.benchmark")


@torch.no_grad()
def measure_inference(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    warmup_iters: int,
    measure_iters: int,
    input_size: int = 512,
) -> Dict[str, Any]:
    model.eval().to(device)
    x = torch.randn(batch_size, 3, input_size, input_size, device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup_iters):
        if device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(x)
            torch.cuda.synchronize(device)
        else:
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: List[float] = []
    for _ in range(measure_iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(x)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    avg_ms = float(arr.mean())
    med_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))
    std_ms = float(arr.std())
    throughput = float(batch_size / (avg_ms / 1000.0)) if avg_ms > 0 else 0.0

    peak_alloc_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device.type == "cuda" else None
    )
    peak_reserved_mb = (
        round(torch.cuda.max_memory_reserved(device) / (1024**2), 2) if device.type == "cuda" else None
    )

    return {
        "device": str(device),
        "precision": "fp16_amp" if (device.type == "cuda" and use_amp) else "fp32",
        "batch_size": int(batch_size),
        "input_shape": [int(batch_size), 3, input_size, input_size],
        "warmup_iterations": int(warmup_iters),
        "measured_iterations": int(measure_iters),
        "avg_latency_ms": round(avg_ms, 3),
        "median_latency_ms": round(med_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "std_latency_ms": round(std_ms, 3),
        "avg_ms_per_image": round(avg_ms / batch_size, 3),
        "throughput_images_per_sec": round(throughput, 3),
        "peak_gpu_memory_allocated_mb": peak_alloc_mb,
        "peak_gpu_memory_reserved_mb": peak_reserved_mb,
    }


def benchmark_all() -> Dict[str, Any]:
    root = get_project_root()
    has_cuda = torch.cuda.is_available()
    env_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cpu_threads_used": torch.get_num_threads(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cuda_available": has_cuda,
        "gpu_name": torch.cuda.get_device_name(0) if has_cuda else None,
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if has_cuda else None,
        "input_resolution": "512x512 RGB",
    }

    specs = [
        ("unet", "U-Net + ResNet34", UNetResNet34, root / "experiments" / "unet" / "best_model.pth"),
        ("deeplabv3plus", "DeepLabV3+ + ResNet50", DeepLabV3PlusResNet50, root / "experiments" / "deeplabv3plus" / "best_model.pth"),
    ]

    results: Dict[str, Any] = {"environment": env_info, "models": {}}
    cpu_dev = torch.device("cpu")
    gpu_dev = torch.device("cuda") if has_cuda else None

    for key, label, cls, ckpt_path in specs:
        LOGGER.info("Benchmarking %s ...", label)
        model = cls(num_classes=4, pretrained=False)
        load_checkpoint(ckpt_path, model, cpu_dev)

        total_params = int(sum(p.numel() for p in model.parameters()))
        trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        state_bytes = int(sum(t.numel() * t.element_size() for t in model.state_dict().values()))
        ckpt_bytes = int(ckpt_path.stat().st_size)

        entry: Dict[str, Any] = {
            "architecture": label,
            "parameter_count": total_params,
            "trainable_parameters": trainable_params,
            "weights_only_size_mb": round(state_bytes / (1024**2), 2),
            "checkpoint_path": str(ckpt_path.relative_to(root)).replace("\\", "/"),
            "checkpoint_size_bytes": ckpt_bytes,
            "checkpoint_size_mb": round(ckpt_bytes / (1024**2), 2),
            "profiles": {},
        }

        if gpu_dev is not None:
            entry["profiles"]["gpu_fp16_bs1"] = measure_inference(
                model, gpu_dev, batch_size=1, use_amp=True, warmup_iters=20, measure_iters=100
            )
            entry["profiles"]["gpu_fp16_bs4"] = measure_inference(
                model, gpu_dev, batch_size=4, use_amp=True, warmup_iters=15, measure_iters=60
            )
            entry["profiles"]["gpu_fp32_bs1"] = measure_inference(
                model, gpu_dev, batch_size=1, use_amp=False, warmup_iters=15, measure_iters=60
            )
            model.to(cpu_dev)
            torch.cuda.empty_cache()

        entry["profiles"]["cpu_fp32_bs1"] = measure_inference(
            model, cpu_dev, batch_size=1, use_amp=False, warmup_iters=5, measure_iters=20
        )
        results["models"][key] = entry

    out_path = root / "experiments" / "evaluation" / "benchmarks" / "inference_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved hardware benchmark report to %s", out_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    benchmark_all()

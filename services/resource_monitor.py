"""Process resource snapshots for evaluation and inspector telemetry."""

from __future__ import annotations

from typing import Any


def snapshot_resources() -> dict[str, Any]:
    try:
        import os

        import psutil

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        vm = psutil.virtual_memory()
        cpu = proc.cpu_percent(interval=0.05)
        gpu: dict[str, Any] = {"available": False}
        try:
            import shutil
            import subprocess

            if shutil.which("nvidia-smi"):
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=2,
                    text=True,
                )
                line = out.strip().splitlines()[0] if out.strip() else ""
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpu = {
                        "available": True,
                        "name": parts[0],
                        "utilization_percent": float(parts[1]),
                        "memory_used_mb": float(parts[2]),
                        "memory_total_mb": float(parts[3]),
                    }
        except Exception:
            pass
        return {
            "rss_mb": round(mem.rss / (1024 * 1024), 2),
            "vms_mb": round(mem.vms / (1024 * 1024), 2),
            "process_cpu_percent": cpu,
            "system_memory_percent": vm.percent,
            "gpu": gpu,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

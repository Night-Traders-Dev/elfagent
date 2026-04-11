from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
from pathlib import Path


def _read_meminfo() -> dict[str, int]:
    meminfo: dict[str, int] = {}
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return meminfo
    for line in meminfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        number = value.strip().split()[0]
        if number.isdigit():
            meminfo[key] = int(number)
    return meminfo


def system_info() -> str:
    """Return basic host information for the current development machine."""
    disk = shutil.disk_usage(Path.cwd())
    meminfo = _read_meminfo()
    top_processes = ""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,comm,%cpu,%mem", "--sort=-%cpu"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        top_processes = "\n".join((result.stdout or "").splitlines()[:6]).strip()
    except Exception:
        top_processes = "ps output unavailable"

    load_average = "unavailable"
    if hasattr(os, "getloadavg"):
        try:
            load_average = ", ".join(f"{value:.2f}" for value in os.getloadavg())
        except OSError:
            pass

    mem_total_mb = round(meminfo.get("MemTotal", 0) / 1024, 1)
    mem_available_mb = round(meminfo.get("MemAvailable", 0) / 1024, 1)
    return (
        f"Host: {platform.node()}\n"
        f"Platform: {platform.platform()}\n"
        f"Python: {platform.python_version()}\n"
        f"CPU cores: {os.cpu_count()}\n"
        f"Load average: {load_average}\n"
        f"Memory: total={mem_total_mb} MiB available={mem_available_mb} MiB\n"
        f"Disk ({Path.cwd()}): total={round(disk.total / 1024**3, 2)} GiB "
        f"used={round((disk.total - disk.free) / 1024**3, 2)} GiB "
        f"free={round(disk.free / 1024**3, 2)} GiB\n\n"
        f"Top processes:\n{top_processes}"
    )


def list_serial_ports() -> str:
    """List likely serial devices on the current host."""
    candidates = [
        "/dev/serial/by-id/*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "/dev/ttyS*",
        "/dev/cu.*",
    ]
    ports: list[str] = []
    for pattern in candidates:
        for item in sorted(glob.glob(pattern)):
            if item not in ports:
                ports.append(item)
    if not ports:
        return "No serial ports detected."
    return "Serial ports:\n" + "\n".join(f"- {port}" for port in ports)

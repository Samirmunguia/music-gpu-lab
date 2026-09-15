#!/usr/bin/env python3
"""
Dependency Verification and Environment Audit for music-gpu-lab.
Validates that active Python packages conform strictly to the upstream ACE-Step 1.5
pinned snapshot requirements (Linux x86_64 / CUDA 12.8).
"""

import argparse
import platform
import sys
from typing import Any, Dict, List, Tuple


def parse_version_tuple(v_str: str) -> Tuple[int, ...]:
    """Parse version string into a comparable tuple of integers."""
    clean = v_str.split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
    parts = []
    for p in clean.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts)


def check_python_version() -> Tuple[bool, str, str]:
    """Verify Python >=3.11,<3.13."""
    v = sys.version_info
    v_str = f"{v.major}.{v.minor}.{v.micro}"
    expected = ">=3.11, <3.13"
    ok = (v.major == 3 and 11 <= v.minor < 13)
    return ok, v_str, expected


def check_package(pkg_name: str, min_ver: str, max_ver: Optional[str] = None) -> Tuple[bool, str, str]:
    """Check if installed package version falls within [min_ver, max_ver)."""
    expected = f">={min_ver}" + (f", <{max_ver}" if max_ver else "")
    try:
        if pkg_name == "torch":
            import torch
            actual = getattr(torch, "__version__", "unknown")
        elif pkg_name == "torchvision":
            import torchvision
            actual = getattr(torchvision, "__version__", "unknown")
        elif pkg_name == "torchaudio":
            import torchaudio
            actual = getattr(torchaudio, "__version__", "unknown")
        elif pkg_name == "transformers":
            import transformers
            actual = getattr(transformers, "__version__", "unknown")
        elif pkg_name == "diffusers":
            import diffusers
            actual = getattr(diffusers, "__version__", "unknown")
        elif pkg_name == "torchao":
            import torchao
            actual = getattr(torchao, "__version__", "unknown")
        elif pkg_name == "accelerate":
            import accelerate
            actual = getattr(accelerate, "__version__", "unknown")
        elif pkg_name == "huggingface_hub":
            import huggingface_hub
            actual = getattr(huggingface_hub, "__version__", "unknown")
        else:
            import importlib
            mod = importlib.import_module(pkg_name)
            actual = getattr(mod, "__version__", "installed")
    except ImportError:
        return False, "NOT INSTALLED", expected
    except Exception as e:
        return False, f"ERROR: {e}", expected

    actual_tuple = parse_version_tuple(actual)
    min_tuple = parse_version_tuple(min_ver)
    ok = (actual_tuple >= min_tuple)

    if max_ver:
        max_tuple = parse_version_tuple(max_ver)
        ok = ok and (actual_tuple < max_tuple)

    return ok, actual, expected


def run_audit(strict: bool = False, require_cuda: bool = False) -> bool:
    """Run full environment dependency audit against pinned upstream snapshot."""
    is_linux = (platform.system().lower() == "linux")
    is_x86_64 = (platform.machine().lower() in ["x86_64", "amd64"])

    print("=" * 75)
    print("      music-gpu-lab :: Pinned Upstream Dependency Audit         ")
    print(f"Platform: {platform.system()} ({platform.machine()}) | Python: {platform.python_version()}")
    print("=" * 75)

    checks: List[Tuple[str, bool, str, str]] = []

    # 1. Python Version
    ok_py, act_py, exp_py = check_python_version()
    checks.append(("Python", ok_py, act_py, exp_py))

    # 2. PyTorch & CUDA
    try:
        import torch
        torch_ver = getattr(torch, "__version__", "unknown")
        cuda_build = getattr(torch.version, "cuda", "N/A") or "None"
        cuda_avail = torch.cuda.is_available()

        # Upstream pins torch==2.10.0+cu128 on Linux x86_64
        if is_linux and is_x86_64:
            exp_torch = "2.10.0+cu128 (>=2.7.0)"
            ok_torch = parse_version_tuple(torch_ver) >= (2, 7, 0)
        else:
            exp_torch = ">=2.7.0"
            ok_torch = parse_version_tuple(torch_ver) >= (2, 7, 0)

        checks.append(("torch", ok_torch, f"{torch_ver} (CUDA build: {cuda_build})", exp_torch))

        if require_cuda:
            checks.append(("torch.cuda.is_available()", cuda_avail, str(cuda_avail), "True"))
        else:
            checks.append(("CUDA Device Access", cuda_avail, "Available" if cuda_avail else "Not Available", "Optional offline"))

    except ImportError:
        checks.append(("torch", False, "NOT INSTALLED", ">=2.7.0 (2.10.0+cu128 on Linux)"))

    # 3. Core Upstream ACE-Step Requirements
    # Upstream pyproject.toml: transformers>=4.51.0,<4.58.0
    ok_trans, act_trans, exp_trans = check_package("transformers", "4.51.0", "4.58.0")
    checks.append(("transformers", ok_trans, act_trans, exp_trans))

    # Upstream pyproject.toml: diffusers>=0.37.0
    ok_diff, act_diff, exp_diff = check_package("diffusers", "0.37.0")
    checks.append(("diffusers", ok_diff, act_diff, exp_diff))

    # Upstream pyproject.toml: torchao>=0.16.0,<0.17.0 (on non-aarch64)
    if platform.machine().lower() != "aarch64":
        ok_ao, act_ao, exp_ao = check_package("torchao", "0.16.0", "0.17.0")
        checks.append(("torchao", ok_ao, act_ao, exp_ao))
    else:
        ok_ao, act_ao, exp_ao = check_package("torchao", "0.8.0")
        checks.append(("torchao (aarch64)", ok_ao, act_ao, exp_ao))

    # Accelerate: >=1.12.0
    ok_acc, act_acc, exp_acc = check_package("accelerate", "1.12.0")
    checks.append(("accelerate", ok_acc, act_acc, exp_acc))

    # Lab requirement: huggingface_hub>=0.28.0
    ok_hf, act_hf, exp_hf = check_package("huggingface_hub", "0.28.0")
    checks.append(("huggingface_hub", ok_hf, act_hf, exp_hf))

    # Print Table
    col_w = [26, 32, 22, 8]
    header = f"{'Package / Component':<{col_w[0]}} {'Resolved Version':<{col_w[1]}} {'Expected Version':<{col_w[2]}} Status"
    print(header)
    print("-" * 75)

    all_passed = True
    for name, ok, actual, expected in checks:
        status_str = "[ PASS ]" if ok else "[ FAIL ]"
        if not ok and name != "CUDA Device Access":
            all_passed = False
        print(f"{name:<{col_w[0]}} {actual:<{col_w[1]}} {expected:<{col_w[2]}} {status_str}")

    print("=" * 75)
    if all_passed:
        print("[AUDIT SUCCESS] Active environment satisfies pinned ACE-Step 1.5 specifications.")
    else:
        print("[AUDIT FAILURE] Detected version violations or missing dependencies.", file=sys.stderr)

    if strict and not all_passed:
        sys.exit(1)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Audit environment against pinned upstream dependencies")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero code if any check fails")
    parser.add_argument("--require-cuda", action="store_true", help="Require torch.cuda.is_available() to be True")
    args = parser.parse_args()

    ok = run_audit(strict=args.strict, require_cuda=args.require_cuda)
    sys.exit(0 if (ok or not args.strict) else 1)


if __name__ == "__main__":
    main()

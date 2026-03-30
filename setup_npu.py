#!/usr/bin/env python3
"""
NPU Setup Script for Rock 5B (RK3588)
======================================
Run this ONCE to:
  1. Check if NPU is available
  2. Install required packages
  3. Export trained ML models to ONNX (ready for RKNN conversion)
  4. Print instructions for completing the NPU setup

Usage:
    python setup_npu.py
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

def check_npu_driver() -> bool:
    """Check if RK3588 NPU kernel driver is loaded."""
    path = "/sys/kernel/debug/rknpu/version"
    if os.path.exists(path):
        with open(path) as f:
            version = f.read().strip()
        print(f"  ✅ NPU driver found: {version}")
        return True
    print("  ❌ NPU driver not found")
    print("     Make sure you are running Radxa's official Debian image")
    print("     The NPU driver (RKNPU2) is included in Radxa's kernel")
    return False


def check_npu_load():
    """Check current NPU utilization."""
    path = "/sys/kernel/debug/rknpu/load"
    if os.path.exists(path):
        with open(path) as f:
            load = f.read().strip()
        print(f"  📊 NPU load: {load}")
    else:
        print("  📊 NPU load: unavailable")


def install_rknn_lite():
    """Install RKNN Lite Python package."""
    print("\n[2] Installing RKNN Lite...")
    try:
        import rknnlite
        print("  ✅ rknn-toolkit-lite2 already installed")
        return True
    except ImportError:
        pass

    # Try pip install
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "rknn-toolkit-lite2"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  ✅ rknn-toolkit-lite2 installed")
        return True
    else:
        print(f"  ⚠️  pip install failed: {result.stderr[:200]}")
        print("  Try manually: pip install rknn-toolkit-lite2")
        return False


def install_onnx():
    """Install ONNX and skl2onnx for model export."""
    print("\n[3] Installing ONNX packages...")
    packages = ["onnx", "onnxruntime", "skl2onnx"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ✅ {pkg} already installed")
        except ImportError:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  ✅ {pkg} installed")
            else:
                print(f"  ⚠️  {pkg} failed: {result.stderr[:100]}")


def export_ml_models():
    """Export trained ML models to ONNX format."""
    print("\n[4] Exporting ML models to ONNX...")

    sys.path.insert(0, str(ROOT))

    try:
        from src.ops.npu_accelerator import ModelExporter
        exporter = ModelExporter("data_cache/npu_models")

        # Try to load and export MLStrategy model
        try:
            from src.strategy.ml_strategy import MLStrategy
            from src.data.market_data import MarketDataFetcher
            import yfinance as yf

            print("  Training MLStrategy on SPY for export...")
            df = yf.download("SPY", period="3y", progress=False)
            if df.empty:
                print("  ⚠️  Could not download training data")
                return

            ml = MLStrategy()
            ml.train(df)

            if ml._model is not None:
                path = exporter.export_ml_strategy(ml._model, n_features=16)
                if path:
                    print(f"  ✅ MLStrategy exported: {path}")
            else:
                print("  ⚠️  MLStrategy model not trained")

        except Exception as e:
            print(f"  ⚠️  MLStrategy export failed: {e}")

        # Generate conversion script
        script = exporter.generate_conversion_script()
        print(f"  ✅ RKNN conversion script saved: {script}")

    except Exception as e:
        print(f"  ⚠️  Model export failed: {e}")


def create_model_dir():
    """Create NPU model directory."""
    model_dir = ROOT / "data_cache" / "npu_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ NPU model directory: {model_dir}")
    return model_dir


def print_next_steps(model_dir: Path, npu_available: bool):
    """Print instructions for completing NPU setup."""
    print("\n" + "═" * 60)
    print("  NPU SETUP — NEXT STEPS")
    print("═" * 60)

    if npu_available:
        print("""
✅ NPU is available on your Rock 5B!

To fully enable NPU acceleration for FinBERT and ML models:

STEP 1 — Copy .onnx files to your x86_64 PC:
  scp rock@192.168.1.34:~/tmp/Traide/Trade.1.0/alpha-trading-platform-main/data_cache/npu_models/*.onnx .

STEP 2 — On your x86_64 PC, install rknn-toolkit2:
  pip install rknn-toolkit2

STEP 3 — Run the conversion script on your x86_64 PC:
  python convert_to_rknn.py

STEP 4 — Copy .rknn files back to Rock 5B:
  scp *.rknn rock@192.168.1.34:~/tmp/Traide/Trade.1.0/alpha-trading-platform-main/data_cache/npu_models/

STEP 5 — Restart the platform:
  ~/start_trader.sh

The platform will automatically detect the .rknn files and use the NPU.

CURRENT STATUS (without .rknn files):
  - Sentiment analysis: keyword fallback (instant, lower quality)
  - ML inference: sklearn CPU (slow but works)

AFTER NPU SETUP:
  - Sentiment analysis: FinBERT on NPU (~10-15ms per article)
  - ML inference: RKNN on NPU (~2-5ms per prediction)
""")
    else:
        print("""
❌ NPU driver not detected.

Make sure you are running Radxa's official Debian image for Rock 5B.
The NPU driver (RKNPU2) is included in Radxa's kernel.

Without NPU, the platform still works with CPU fallback:
  - Sentiment: keyword-based (fast, lower quality)
  - ML models: sklearn on CPU (normal speed)

To check NPU driver:
  sudo cat /sys/kernel/debug/rknpu/version
""")

    print("═" * 60)


def main():
    print("Rock 5B NPU Setup for Alpha Trading Platform")
    print("=" * 60)

    # 1. Check NPU driver
    print("\n[1] Checking NPU availability...")
    npu_available = check_npu_driver()
    check_npu_load()

    # 2. Install RKNN Lite
    install_rknn_lite()

    # 3. Install ONNX packages
    install_onnx()

    # 4. Create model directory
    print("\n[4] Setting up model directory...")
    model_dir = create_model_dir()

    # 5. Export ML models
    export_ml_models()

    # 6. Print next steps
    print_next_steps(model_dir, npu_available)


if __name__ == "__main__":
    main()

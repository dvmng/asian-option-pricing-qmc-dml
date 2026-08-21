from __future__ import annotations

import sys

import torch


def main():
    print(f"Python:                 {sys.version.split()[0]}")
    print(f"PyTorch:                {torch.__version__}")
    print(f"CUDA available:         {torch.cuda.is_available()}")
    print(f"PyTorch CUDA runtime:   {torch.version.cuda}")

    if not torch.cuda.is_available():
        print(
            "\nCUDA is NOT available to PyTorch.\n"
            "Your RTX 4060 will not be used until a CUDA-enabled PyTorch build "
            "is installed in this .venv."
        )
        raise SystemExit(1)

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)

    print(f"CUDA device index:      {idx}")
    print(f"GPU:                    {props.name}")
    print(f"Compute capability:     {props.major}.{props.minor}")
    print(f"VRAM:                   {props.total_memory / 1024**3:.2f} GB")

    # Small functional CUDA test.
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()

    print(f"Tensor device:          {y.device}")
    print(f"CUDA functional test:   OK")
    print(f"Allocated VRAM now:     {torch.cuda.memory_allocated() / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()

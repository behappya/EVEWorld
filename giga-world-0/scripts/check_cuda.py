import sys

import torch


def main():
    print("torch:", torch.__version__)
    print("compiled cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    try:
        torch.cuda.init()
    except RuntimeError:
        print("", file=sys.stderr)
        print("CUDA initialization failed.", file=sys.stderr)
        print("This usually means the PyTorch CUDA wheel is newer than the node driver.", file=sys.stderr)
        print("For this cluster, install the CUDA 12.8 stack with:", file=sys.stderr)
        print("  cd giga-world-0", file=sys.stderr)
        print("  ./scripts/install_torch_cuda128.sh", file=sys.stderr)
        print("", file=sys.stderr)
        raise
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this job.")
    print("cuda initialized: True")
    print("cuda device 0:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()

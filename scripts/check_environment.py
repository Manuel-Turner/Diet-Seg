import platform


def optional_version(module_name):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "installed")
    except ImportError:
        return "not installed"


def main():
    try:
        import torch
    except ImportError:
        torch = None

    print(f"Python: {platform.python_version()}")
    if torch is None:
        print("PyTorch: not installed")
        print("torch.version.cuda: unavailable")
        print("cuDNN: unavailable")
        print("CUDA available: unavailable")
        print("GPU name: unavailable")
    else:
        print(f"PyTorch: {torch.__version__}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"MONAI: {optional_version('monai')}")
    print(f"NumPy: {optional_version('numpy')}")
    print(f"SciPy: {optional_version('scipy')}")
    print(f"nibabel: {optional_version('nibabel')}")
    print(f"SimpleITK: {optional_version('SimpleITK')}")


if __name__ == "__main__":
    main()

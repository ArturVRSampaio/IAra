import platform

from gpt4all._pyllmodel import _load_cuda

if platform.system() in ("Linux", "Windows"):
    try:
        from nvidia import cuda_runtime, cublas
    except ImportError:
        pass  # CUDA is optional
    else:
        for rtver, blasver in [("12", "12"), ("11.0", "11")]:
            try:
                _load_cuda(rtver, blasver)
                cuda_found = True
            except OSError:  # dlopen() does not give specific error codes
                pass  # try the next one

        print(cuda_found)

def is_nvtabular_available():
    try:
        import nvtabular
    except ImportError:
        nvtabular = None
    return nvtabular is not None


def is_cudf_available():
    try:
        import cudf
        import cupy
    except ImportError:
        cudf = None
        cupy = None
    return cudf is not None and cupy is not None


def is_pyarrow_available():
    try:
        import pyarrow
    except ImportError:
        pyarrow = None
    return pyarrow is not None

from typing import Annotated

import numpy
from numpy.typing import NDArray


class memoryProps:
    @property
    def l2CacheSize(self) -> int: ...

    @property
    def regsPerBlock(self) -> int: ...

    @property
    def regsPerMultiprocessor(self) -> int: ...

    @property
    def memoryBusWidth(self) -> int: ...

    @property
    def memory_clock_rate(self) -> int: ...

    @property
    def sharedMemPerBlock(self) -> int: ...

    @property
    def sharedMemPerMultiprocessor(self) -> int: ...

    @property
    def totalGlobalMem(self) -> int: ...

    @property
    def memory_band_width(self) -> float: ...

class computeProps:
    @property
    def maxGridSize(self) -> "dim3": ...

    @property
    def maxThreadsDim(self) -> "dim3": ...

    @property
    def maxThreadsPerBlock(self) -> int: ...

    @property
    def maxBlocksPerMultiProcessor(self) -> int: ...

    @property
    def maxThreadsPerMultiProcessor(self) -> int: ...

    @property
    def multiProcessorCount(self) -> int: ...

    @property
    def core_clock_rate(self) -> int: ...

class deviceProps:
    def __init__(self) -> None: ...

    @property
    def device_ID(self) -> int: ...

    @property
    def name(self) -> "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >": ...

    @property
    def memoryProps(self) -> memoryProps: ...

    @property
    def computeProps(self) -> computeProps: ...

def get_device_propertity(deviceId: int = 0) -> deviceProps:
    """Get cuda device property"""

def print_device_properties(deviceId: int = 0) -> None:
    """Print cuda device property"""

def sgemm_naive(M: int, N: int, K: int, alpha: float, A: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], B: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], beta: float, C: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda')]) -> None:
    """A naive SGEMM implementation calling CUDA code"""

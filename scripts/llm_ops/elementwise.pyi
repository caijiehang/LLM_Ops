"""
An element-wise operation operates on corresponding elements between tensors.
"""

from typing import Annotated

import numpy
from numpy.typing import NDArray


def elementwise_f32(a: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], b: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], c: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], N: int) -> None: ...

def elementwise_f32x4(a: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], b: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], c: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], N: int) -> None: ...

def elementwise_f16(a: Annotated[NDArray, dict(shape=(None,), order='C', device='cuda')], b: Annotated[NDArray, dict(shape=(None,), order='C', device='cuda')], c: Annotated[NDArray, dict(shape=(None,), order='C', device='cuda')], N: int) -> None: ...

def elementwise_f16x2(a: Annotated[NDArray, dict(shape=(None,), order='C', device='cuda')], b: Annotated[NDArray, dict(shape=(None,), order='C', device='cuda')], c: Annotated[NDArray, dict(shape=(None,), order='C', device='cuda')], N: int) -> None: ...

def elementwise_f16x8_pack(a: Annotated[NDArray, dict(shape=(None,), order='C', device='cuda')], b: Annotated[NDArray, dict(shape=(None,), order='C', device='cuda')], c: Annotated[NDArray, dict(shape=(None,), order='C', device='cuda')], N: int) -> None: ...

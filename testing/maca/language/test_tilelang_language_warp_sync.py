import tilelang
import tilelang.language as T
import torch
from tvm import tir
import tilelang.testing


@tilelang.jit
def kernel_with_warp_sync():
    @T.prim_func
    def main(
        A: T.Tensor((1,), "int32"),
        B: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            if tx == 0:
                tir.call_extern("void", "__nanosleep", 100)
                A[0] = -1
            T.sync_warp()
            if tx == 1:
                B[0] = A[0]

    return main


def test_warp_sync():
    a = torch.empty((1), device="cuda", dtype=torch.int32)
    b = torch.empty((1), device="cuda", dtype=torch.int32)
    kernel = kernel_with_warp_sync()
    assert "__syncwarp" in kernel.get_kernel_source()
    kernel(a, b)
    assert b[0] == -1


@tilelang.jit
def kernel_with_shfl_sync():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            val = tx * 10
            broadcast = T.shfl_sync(val, 31, mask=0xFFFFFFFF)
            A[tx] = broadcast

    return main


def test_shfl_sync():
    a = torch.empty((64), device="cuda", dtype=torch.int32)
    kernel = kernel_with_shfl_sync()
    assert "__shfl_sync" in kernel.get_kernel_source()
    kernel(a)
    assert torch.all(a == 310)


@tilelang.jit
def kernel_with_shfl_up():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            val = tx
            res = T.shfl_up(val, 1, mask=0xFFFFFFFF)
            A[tx] = res

    return main


def test_shfl_up():
    a = torch.empty((64), device="cuda", dtype=torch.int32)
    kernel = kernel_with_shfl_up()
    assert "__shfl_up" in kernel.get_kernel_source()
    kernel(a)
    assert a[0] == 0
    for i in range(1, 64):
        assert a[i] == i - 1


@tilelang.jit
def kernel_with_shfl_down():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            val = tx
            res = T.shfl_down(val, 1, mask=0xFFFFFFFF)
            A[tx] = res

    return main


def test_shfl_down():
    a = torch.empty((64), device="cuda", dtype=torch.int32)
    kernel = kernel_with_shfl_down()
    assert "__shfl_down" in kernel.get_kernel_source()
    kernel(a)
    assert a[63] == 63
    for i in range(63):
        assert a[i] == i + 1


@tilelang.jit
def kernel_with_shfl_xor():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            val = tx
            res = T.shfl_xor(val, 1, mask=0xFFFFFFFF)
            A[tx] = res

    return main


def test_shfl_xor():
    a = torch.empty((64), device="cuda", dtype=torch.int32)
    kernel = kernel_with_shfl_xor()
    assert "__shfl_xor" in kernel.get_kernel_source()
    kernel(a)
    for i in range(64):
        assert a[i] == i ^ 1


if __name__ == "__main__":
    tilelang.testing.main()

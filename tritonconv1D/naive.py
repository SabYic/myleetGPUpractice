import torch
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input, kernel, output,
    input_size, kernel_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # 本 program 负责的一段输出下标
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    o_mask = offs < input_size-kernel_size+1

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # y[o] = sum_k x[o+k] * w[k]
    for k in range(0, kernel_size):
        ix   = offs + k
        xval = tl.load(input + ix, mask=o_mask, other=0.0)
        wval = tl.load(kernel + k)   # 标量
        acc += xval * wval
    tl.store(output + offs, acc, mask=o_mask)

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 512
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)
    
    conv1d_kernel[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )
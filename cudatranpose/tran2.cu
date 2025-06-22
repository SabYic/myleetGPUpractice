#include "solve.h"
#include <cuda_runtime.h>
// best performance 1.86 ms(tesla T4) beat 97
#define BLOCK_SIZE 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 避免 bank conflict

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;  
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;  

    // 读入共享内存：按 input[row][col] 读
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // 写出转置后的位置：按 output[col][row] 写,这里后面是x对x，y对y
    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    // 合并访问全局内存
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
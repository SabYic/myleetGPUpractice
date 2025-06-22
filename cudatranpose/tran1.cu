#include "solve.h"
#include <cuda_runtime.h>
// best performance 2.7986ms(tesla T4) beat 43
#define BLOCK_SIZE 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE ]; 
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;  
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;  

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.y][threadIdx.x];
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
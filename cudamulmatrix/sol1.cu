//best performance 692ms(tesla T4) beat 64.5
//naive maxmul
#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行号
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列号

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];  // A[row][i] * B[i][col]
        }
        C[row * K + col] = sum;  // C[row][col] = sum
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

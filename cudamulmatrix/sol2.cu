//best performance 575ms(tesla T4) beat 81
//tiled maxmul
#include "solve.h"
#include <cuda_runtime.h>
#define TS 32
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int row = threadIdx.y;  // 本地 tile 中的 row 索引
    const int col = threadIdx.x;  // 本地 tile 中的 col 索引

    const int globalRow = blockIdx.y * TS + row;  // 输出矩阵 C 的全局行号
    const int globalCol = blockIdx.x * TS + col;  // 输出矩阵 C 的全局列号

    // Allocate shared memory for A 和 B 的 tile
    __shared__ float Asub[TS][TS];
    __shared__ float Bsub[TS][TS];

    // 累加器初始化
    float acc = 0.0f;

    // 遍历 A 和 B 的 K 方向上的 tile
    const int numTiles = (N+ TS - 1) / TS;

    for (int t = 0; t < numTiles; ++t) {

        // 计算当前 tile 中的 A 和 B 的访问索引
        int tiledRow = t * TS + row;
        int tiledCol = t * TS + col;

        // 加载 A 和 B 的 tile（行主序：A[M][K], B[K][N]）
        Asub[row][col] = (globalRow < M && tiledCol < N) ? A[globalRow * N + tiledCol] : 0.0f;
        Bsub[row][col] = (tiledRow < N && globalCol < K) ? B[tiledRow * K + globalCol] : 0.0f;

        // 同步线程，确保 tile 加载完毕
        __syncthreads();

        // 进行 TS 次乘加（tile 内部的计算）
        for (int k = 0; k < TS; ++k) {
            acc += Asub[row][k] * Bsub[k][col];
        }

        // 同步线程，确保当前 tile 的计算结束
        __syncthreads();
    }

    // 写回结果
    if (globalRow < M && globalCol < K) {
        C[globalRow * K + globalCol] = acc;
    }
}
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TS, TS);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

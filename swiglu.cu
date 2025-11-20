#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int tid = threadIdx.x+(blockDim.x*blockIdx.x);
    float inp,inp2;
    for(int i=tid;i<halfN;i+=(gridDim.x*blockDim.x)){
        inp = input[i];
        inp2 = input[i+halfN];
        inp = inp /(1.0f + __expf(-inp));
        inp *= inp2;
        output[i]=inp; 
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 512;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
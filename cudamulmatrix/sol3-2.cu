
#define floatX float2
#define WIDTH 2
#define TS 32
///////////////////////////////
// 🟢 改进版本（性能更好） best performance 418ms(tesla T4) beat 92.4
///////////////////////////////
__global__ void matmul_float2_fast(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int globalRow = blockIdx.y * TS + ty;
    int globalCol = (blockIdx.x * (TS / WIDTH) + tx) * WIDTH;

    __shared__ floatX Asub[TS][TS / WIDTH];
    __shared__ floatX Bsub[TS][TS / WIDTH];

    floatX acc = make_float2(0.f, 0.f);
    int numTiles = (K + TS - 1) / TS;

    for (int t = 0; t < numTiles; ++t) {
        int tiledRow = t * TS + ty;
        int tiledCol = (t * TS + tx * WIDTH);

        float valA0 = 0.f, valA1 = 0.f;
        if (globalRow < M) {
            if (tiledCol + 0 < K) valA0 = A[globalRow * K + tiledCol + 0];
            if (tiledCol + 1 < K) valA1 = A[globalRow * K + tiledCol + 1];
        }
        Asub[ty][tx] = make_float2(valA0, valA1);

        float valB0 = 0.f, valB1 = 0.f;
        if (tiledRow < K) {
            if (globalCol + 0 < N) valB0 = B[tiledRow * N + globalCol + 0];
            if (globalCol + 1 < N) valB1 = B[tiledRow * N + globalCol + 1];
        }
        Bsub[ty][tx] = make_float2(valB0, valB1);

        __syncthreads();

        for (int k = 0; k < TS / WIDTH; ++k) {
            floatX vecB = Bsub[k][tx];
            for (int w = 0; w < WIDTH; ++w) {
                floatX vecA = Asub[WIDTH * k + w][ty];
                float valB = (w == 0) ? vecB.x : vecB.y;
                acc.x += vecA.x * valB;
                acc.y += vecA.y * valB;
            }
        }

        __syncthreads();
    }

    if (globalRow < M) {
        if (globalCol + 0 < N) C[globalRow * N + globalCol + 0] = acc.x;
        if (globalCol + 1 < N) C[globalRow * N + globalCol + 1] = acc.y;
    }
}

// 对比：你的原始代码（慢） VS 改进后的代码（快）应该是float8的时候最快
// vectorized tiled maxmul GPT改进 没改进应该是load store没合并的问题
///////////////////////////////
// 🔴 原始版本（性能较差）
///////////////////////////////
#define floatX float2
#define WIDTH 2
#define TS 32
__global__ void matmul_float2_slow(const float* A, const float* B, float* C, int M, int N, int K) {

    int row_tile = threadIdx.y;
    int col_tile = threadIdx.x;

    int globalRow = blockIdx.y * TS + row_tile;
    int globalCol = blockIdx.x * TS / WIDTH + col_tile;

    __shared__ floatX Asub[TS][TS / WIDTH];
    __shared__ floatX Bsub[TS][TS / WIDTH];

    floatX acc = make_float2(0.f, 0.f);

    int numTiles = ((K + TS / WIDTH - 1) / (TS / WIDTH));

    for (int t = 0; t < numTiles; ++t) {
        int tiledrow = TS * t + row_tile;
        int tiledcol = TS / WIDTH * t + col_tile;

        float valA0 = 0.f, valA1 = 0.f;
        if (globalRow < M) {
            valA0 = (tiledcol * WIDTH < K) ? A[globalRow * K + tiledcol * WIDTH] : 0.0f;
            valA1 = (tiledcol * WIDTH + 1 < K) ? A[globalRow * K + tiledcol * WIDTH + 1] : 0.0f;
        }
        Asub[row_tile][col_tile] = make_float2(valA0, valA1);

        float valB0 = 0.f, valB1 = 0.f;
        if (globalCol * WIDTH < N && tiledrow < K)
            valB0 = B[tiledrow * N + globalCol * WIDTH];
        if (globalCol * WIDTH + 1 < N && tiledrow < K)
            valB1 = B[tiledrow * N + globalCol * WIDTH + 1];
        Bsub[row_tile][col_tile] = make_float2(valB0, valB1);

        __syncthreads();

        for (int k = 0; k < TS / WIDTH; ++k) {
            floatX vecA = Asub[row_tile][k];
            for (int w = 0; w < WIDTH; w++) {
                floatX vecB = Bsub[WIDTH * k + w][col_tile];
                float valA = (w == 0) ? vecA.x : vecA.y;
                acc.x += vecB.x * valA;
                acc.y += vecB.y * valA;
            }
        }
        __syncthreads();
    }

    if (globalRow < M) {
        if (globalCol * WIDTH + 0 < N) C[globalRow * N + globalCol * WIDTH + 0] = acc.x;
        if (globalCol * WIDTH + 1 < N) C[globalRow * N + globalCol * WIDTH + 1] = acc.y;
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 32  
#define BLOCK_SIZE 32
#define INPUT_VEC 32
#define BATCH_SIZE 32
#define OUTPUT_VEC 32
#define STREAMS 4
#define LAYERS 4
#define LEARNING_RATE 0.001f
#define ITERATIONS 100

// Forward kernels

__global__ void matmul_fused_bias_shared_mem_tiling(float *A, float *B, float *C, float *bias, int m, int k, int n){

    __shared__ float tileA [BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float tileB [BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float tileBias[BLOCK_SIZE];

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (k + BLOCK_SIZE - 1)/BLOCK_SIZE;

    if(localRow == 0) tileBias[localCol] = (globalCol < n) ? bias[globalCol] : 0.0f;

    __syncthreads();

    for(int i = 0; i < numTiles; i++){

        int tileRow = i * BLOCK_SIZE + localRow;
        int tileCol = i * BLOCK_SIZE + localCol;

        tileA[localRow * BLOCK_SIZE + localCol] = (globalRow < m && tileCol < k) ? A[(globalRow) * k + tileCol] : 0.0f;
        tileB[localRow * BLOCK_SIZE + localCol] = (tileRow < k && globalCol < n) ? B[(tileRow) * n + globalCol] : 0.0f;

        __syncthreads();

        for (int tile_k = 0; tile_k < BLOCK_SIZE; tile_k++){
            sum += tileA[localRow * BLOCK_SIZE + tile_k] * tileB[tile_k * BLOCK_SIZE + localCol];
        }

        __syncthreads();

    }

    if(globalRow < m && globalCol < n){
        C[globalRow * n + globalCol] = sum + tileBias[localCol];
    }

}

__global__ void relu_kernel(float *out, float *mask, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && mask[idx] <= 0.0f)
        out[idx] = 0.0f;
}

// Backward kernels 

// General tiled matmul for the backward pass. Output C is always (m × n);
// the reduction dimension is k.
//   transA=true  → A is stored as (k × m), access A[tileCol * m + globalRow]
//   transB=true  → B is stored as (n × k), access B[globalCol * k + tileRow]
__global__ void matmul_backward(float *A, float *B, float *C,
                                 int m, int k, int n,
                                 bool transA, bool transB) {
    __shared__ float tileA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE * BLOCK_SIZE];

    int localRow = threadIdx.y, localCol = threadIdx.x;
    int globalRow = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < numTiles; i++) {
        int tileRow = i * BLOCK_SIZE + localRow;
        int tileCol = i * BLOCK_SIZE + localCol;

        tileA[localRow * BLOCK_SIZE + localCol] = (globalRow < m && tileCol < k)
            ? (transA ? A[tileCol * m + globalRow] : A[globalRow * k + tileCol])
            : 0.0f;

        tileB[localRow * BLOCK_SIZE + localCol] = (tileRow < k && globalCol < n)
            ? (transB ? B[globalCol * k + tileRow] : B[tileRow * n + globalCol])
            : 0.0f;

        __syncthreads();
        for (int tile_k = 0; tile_k < BLOCK_SIZE; tile_k++)
            sum += tileA[localRow * BLOCK_SIZE + tile_k] * tileB[tile_k * BLOCK_SIZE + localCol];
        __syncthreads();
    }

    if (globalRow < m && globalCol < n)
        C[globalRow * n + globalCol] = sum;
}

// dZ = 2*(Z_out - y) / |S|  where |S| = BATCH_SIZE * OUTPUT_VEC
__global__ void mse_grad(float *Z_out, float *y, float *dZ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dZ[idx] = 2.0f * (Z_out[idx] - y[idx]) / size;
}

// d_b_grad[j] = sum_i d_op_grad[i][j]  (sum over the batch rows)
__global__ void sum_rows_kernel(float *dZ, float *db, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float s = 0.0f;
        for (int r = 0; r < rows; r++) s += dZ[r * cols + col];
        db[col] = s;
    }
}

// W -= lr * dW
__global__ void sgd_update_kernel(float *W, float *dW, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) W[idx] -= LEARNING_RATE * dW[idx];
}

// MISC 

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Forward pass 

void launch_mlp_chunk_forward_pass(float *d_ip_chunk, float **d_w, float **d_b, float **d_op_chunk,
                      int batch_size, cudaStream_t stream){

    dim3 matmul_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 hidden_grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 output_grid((OUTPUT_VEC + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 relu_block(BLOCK_SIZE);
    dim3 relu_grid((batch_size * N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_fused_bias_shared_mem_tiling<<<hidden_grid, matmul_block, 0, stream>>>(
        d_ip_chunk, d_w[0], d_op_chunk[0], d_b[0], batch_size, INPUT_VEC, N);
    relu_kernel<<<relu_grid, relu_block, 0, stream>>>(d_op_chunk[0], d_op_chunk[0], batch_size * N);

    for(int i = 1; i < LAYERS - 1; i++){
        matmul_fused_bias_shared_mem_tiling<<<hidden_grid, matmul_block, 0, stream>>>(
            d_op_chunk[i - 1], d_w[i], d_op_chunk[i], d_b[i], batch_size, N, N);
        relu_kernel<<<relu_grid, relu_block, 0, stream>>>(d_op_chunk[i], d_op_chunk[i], batch_size * N);
    }

    matmul_fused_bias_shared_mem_tiling<<<output_grid, matmul_block, 0, stream>>>(
        d_op_chunk[LAYERS - 2], d_w[LAYERS - 1], d_op_chunk[LAYERS - 1],
        d_b[LAYERS - 1], batch_size, N, OUTPUT_VEC);
}

// Backward pass 
//
// After the forward pass, d_op[l] holds:
//   d_op[0..LAYERS-2] : post-ReLU activations A_1..A_{L-1}   (BATCH_SIZE × N)
//   d_op[LAYERS-1]    : output pre-activation Z_L              (BATCH_SIZE × OUTPUT_VEC)
//
// d_op_grad[l] accumulates the gradient w.r.t. the pre-activation of layer l+1 (dZ).
// Since ReLU is applied in-place, (A > 0) serves as the ReLU mask for the backward pass.
//
// Sequence (all on the default stream):
//   dZ[L-1]   = MSE grad
//   dW[L-1]   = A[L-2]^T @ dZ[L-1]          (transA=true)
//   db[L-1]   = sum_rows(dZ[L-1])
//   dA[L-2]   = dZ[L-1] @ W[L-1]^T          (transB=true) → d_op_grad[L-2]
//   dZ[L-2]   = dA[L-2] * (A[L-2] > 0)      in-place relu backward
//   for l = L-2 down to 1:
//     dW[l]   = A[l-1]^T @ dZ[l]
//     db[l]   = sum_rows(dZ[l])
//     dA[l-1] = dZ[l] @ W[l]^T              → d_op_grad[l-1]
//     dZ[l-1] = dA[l-1] * (A[l-1] > 0)
//   dW[0]     = X^T @ dZ[0]                 (X = d_ip)
//   db[0]     = sum_rows(dZ[0])

void launch_backward(float *d_ip, float **d_w, float **d_op,
                     float **d_w_grad, float **d_b_grad, float **d_op_grad, float *d_y) {
    dim3 blk1d(BLOCK_SIZE);
    dim3 blk2d(BLOCK_SIZE, BLOCK_SIZE);

    // 2D grid covering an (rows × cols) output matrix
    auto g2 = [](int rows, int cols) {
        return dim3((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    };
    auto g1 = [](int sz) { return dim3((sz + BLOCK_SIZE - 1) / BLOCK_SIZE); };

    int out_size = BATCH_SIZE * OUTPUT_VEC;
    int hid_size = BATCH_SIZE * N;

    // Output layer gradient 
    mse_grad<<<g1(out_size), blk1d>>>(d_op[LAYERS-1], d_y, d_op_grad[LAYERS-1], out_size);

    // dW[L-1] = A[L-2]^T @ dZ[L-1]  :  (N×BATCH)@(BATCH×OVEC) → (N×OVEC)
    matmul_backward<<<g2(N, OUTPUT_VEC), blk2d>>>(
        d_op[LAYERS-2], d_op_grad[LAYERS-1], d_w_grad[LAYERS-1],
        N, BATCH_SIZE, OUTPUT_VEC, true, false);

    // db[L-1] = sum_rows(dZ[L-1])
    sum_rows_kernel<<<g1(OUTPUT_VEC), blk1d>>>(
        d_op_grad[LAYERS-1], d_b_grad[LAYERS-1], BATCH_SIZE, OUTPUT_VEC);

    // dA[L-2] = dZ[L-1] @ W[L-1]^T  :  (BATCH×OVEC)@(OVEC×N) → (BATCH×N)
    // W[L-1] is stored (N×OVEC), so transB=true
    matmul_backward<<<g2(BATCH_SIZE, N), blk2d>>>(
        d_op_grad[LAYERS-1], d_w[LAYERS-1], d_op_grad[LAYERS-2],
        BATCH_SIZE, OUTPUT_VEC, N, false, true);

    // dZ[L-2] = dA[L-2] * 1[A[L-2] > 0]
    relu_kernel<<<g1(hid_size), blk1d>>>(d_op_grad[LAYERS-2], d_op[LAYERS-2], hid_size);

    // Hidden layers: l = LAYERS-2 down to 1 
    for (int l = LAYERS - 2; l >= 1; l--) {
        // dW[l] = A[l-1]^T @ dZ[l]  :  (N×BATCH)@(BATCH×N) → (N×N)
        matmul_backward<<<g2(N, N), blk2d>>>(
            d_op[l-1], d_op_grad[l], d_w_grad[l], N, BATCH_SIZE, N, true, false);

        // db[l] = sum_rows(dZ[l])
        sum_rows_kernel<<<g1(N), blk1d>>>(d_op_grad[l], d_b_grad[l], BATCH_SIZE, N);

        // dA[l-1] = dZ[l] @ W[l]^T  :  (BATCH×N)@(N×N) → (BATCH×N)
        matmul_backward<<<g2(BATCH_SIZE, N), blk2d>>>(
            d_op_grad[l], d_w[l], d_op_grad[l-1], BATCH_SIZE, N, N, false, true);

        // dZ[l-1] = dA[l-1] * 1[A[l-1] > 0]
        relu_kernel<<<g1(hid_size), blk1d>>>(d_op_grad[l-1], d_op[l-1], hid_size);
    }

    // First layer: input is X = d_ip  (BATCH_SIZE × INPUT_VEC) 
    // dW[0] = X^T @ dZ[0]  :  (INPUT_VEC×BATCH)@(BATCH×N) → (INPUT_VEC×N)
    matmul_backward<<<g2(INPUT_VEC, N), blk2d>>>(
        d_ip, d_op_grad[0], d_w_grad[0], INPUT_VEC, BATCH_SIZE, N, true, false);

    // db[0] = sum_rows(dZ[0])
    sum_rows_kernel<<<g1(N), blk1d>>>(d_op_grad[0], d_b_grad[0], BATCH_SIZE, N);
}

// SGD update 

void launch_sgd(float **d_w, float **d_b, float **d_w_grad, float **d_b_grad,
                int *n_w, int *n_b) {
    for (int l = 0; l < LAYERS; l++) {
        sgd_update_kernel<<<(n_w[l] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_w[l], d_w_grad[l], n_w[l]);
        sgd_update_kernel<<<(n_b[l] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_b[l], d_b_grad[l], n_b[l]);
    }
}





int main() {

    // check to see if streams are valid or not
    if(BATCH_SIZE % STREAMS != 0){
        printf("Batch size is not divisible by number of streams");
        return 0;
    }

    int chunk = BATCH_SIZE / STREAMS;

    // Memory allocation for forward prop
    float *h_ip, *h_y;
    float *h_w[LAYERS];
    float *h_b[LAYERS];
    float *h_op;

    float *d_ip, *d_y;
    float *d_w[LAYERS];
    float *d_b[LAYERS];
    float *d_op[LAYERS];

    float *d_w_grad[LAYERS];
    float *d_b_grad[LAYERS];
    float *d_op_grad[LAYERS];  

    int size_w[LAYERS];
    int size_b[LAYERS];
    int size_op[LAYERS];
    int size_op_grad[LAYERS];
    int n_w[LAYERS], n_b[LAYERS];  // element counts for SGD

    int size_ip = BATCH_SIZE * INPUT_VEC * sizeof(float);
    int size_y  = BATCH_SIZE * OUTPUT_VEC * sizeof(float);

    // Size calculation
    size_w[0]       = INPUT_VEC * N * sizeof(float);  n_w[0] = INPUT_VEC * N;
    size_b[0]       = N * sizeof(float);              n_b[0] = N;
    size_op[0]      = BATCH_SIZE * N * sizeof(float);
    size_op_grad[0] = BATCH_SIZE * N * sizeof(float);

    for(int i = 1; i < LAYERS - 1; i++){
        size_w[i]       = N * N * sizeof(float);          n_w[i] = N * N;
        size_b[i]       = N * sizeof(float);              n_b[i] = N;
        size_op[i]      = BATCH_SIZE * N * sizeof(float);
        size_op_grad[i] = BATCH_SIZE * N * sizeof(float);
    }

    size_w[LAYERS-1]       = N * OUTPUT_VEC * sizeof(float); n_w[LAYERS-1] = N * OUTPUT_VEC;
    size_b[LAYERS-1]       = OUTPUT_VEC * sizeof(float);     n_b[LAYERS-1] = OUTPUT_VEC;
    size_op[LAYERS-1]      = BATCH_SIZE * OUTPUT_VEC * sizeof(float);
    size_op_grad[LAYERS-1] = BATCH_SIZE * OUTPUT_VEC * sizeof(float);

    // Host malloc & init
    h_ip = (float*)malloc(size_ip);
    h_y  = (float*)malloc(size_y);
    for (int i = 0; i < LAYERS; i++){
        h_w[i] = (float*)malloc(size_w[i]);
        h_b[i] = (float*)malloc(size_b[i]);
    }
    h_op = (float*)malloc(size_op[LAYERS-1]);

    srand(32);
    init_matrix(h_ip, BATCH_SIZE, INPUT_VEC);
    init_matrix(h_y,  BATCH_SIZE, OUTPUT_VEC);
    init_matrix(h_w[0], INPUT_VEC, N);
    init_matrix(h_b[0], 1, N);
    for(int i = 1; i < LAYERS - 1; i++){
        init_matrix(h_w[i], N, N);
        init_matrix(h_b[i], 1, N);
    }
    init_matrix(h_w[LAYERS-1], N, OUTPUT_VEC);
    init_matrix(h_b[LAYERS-1], 1, OUTPUT_VEC);

    // Device malloc
    cudaMalloc(&d_ip, size_ip);
    cudaMalloc(&d_y,  size_y);
    for(int i = 0; i < LAYERS; i++){
        cudaMalloc(&d_w[i],       size_w[i]);
        cudaMalloc(&d_b[i],       size_b[i]);
        cudaMalloc(&d_op[i],      size_op[i]);
        cudaMalloc(&d_w_grad[i],  size_w[i]);
        cudaMalloc(&d_b_grad[i],  size_b[i]);
        cudaMalloc(&d_op_grad[i], size_op_grad[i]);
    }

    // Copy weights & input to device
    cudaMemcpy(d_ip, h_ip, size_ip, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,  h_y,  size_y,  cudaMemcpyHostToDevice);
    for(int i = 0; i < LAYERS; i++){
        cudaMemcpy(d_w[i], h_w[i], size_w[i], cudaMemcpyHostToDevice);
        cudaMemcpy(d_b[i], h_b[i], size_b[i], cudaMemcpyHostToDevice);
        cudaMemset(d_op[i], 0, size_op[i]);
    }

    // Create streams
    cudaStream_t streams[STREAMS];
    for(int s = 0; s < STREAMS; s++) cudaStreamCreate(&streams[s]);

    // CUDA events for per-phase timing
    cudaEvent_t fwd_start, fwd_end, bwd_start, bwd_end;
    cudaEventCreate(&fwd_start);
    cudaEventCreate(&fwd_end);
    cudaEventCreate(&bwd_start);
    cudaEventCreate(&bwd_end);

    float total_fwd_ms = 0.0f, total_bwd_ms = 0.0f;

    // Training loop 
    for (int iter = 0; iter < ITERATIONS; iter++) {

        // Forward pass (multi-stream)
        cudaDeviceSynchronize();
        cudaEventRecord(fwd_start, 0);

        for(int s = 0; s < STREAMS; s++){
            int offset = s * chunk;
            float *d_ip_chunk = d_ip + offset * INPUT_VEC;
            float *d_op_chunk[LAYERS];
            for(int l = 0; l < LAYERS - 1; l++)
                d_op_chunk[l] = d_op[l] + offset * N;
            d_op_chunk[LAYERS-1] = d_op[LAYERS-1] + offset * OUTPUT_VEC;

            launch_mlp_chunk_forward_pass(d_ip_chunk, d_w, d_b, d_op_chunk, chunk, streams[s]);
        }
        for(int s = 0; s < STREAMS; s++) cudaStreamSynchronize(streams[s]);

        cudaEventRecord(fwd_end, 0);
        cudaEventSynchronize(fwd_end);

        float fwd_ms;
        cudaEventElapsedTime(&fwd_ms, fwd_start, fwd_end);
        total_fwd_ms += fwd_ms;

        // Backward pass (default stream)
        cudaEventRecord(bwd_start, 0);
        launch_backward(d_ip, d_w, d_op, d_w_grad, d_b_grad, d_op_grad, d_y);
        cudaEventRecord(bwd_end, 0);
        cudaEventSynchronize(bwd_end);

        float bwd_ms;
        cudaEventElapsedTime(&bwd_ms, bwd_start, bwd_end);
        total_bwd_ms += bwd_ms;

        // SGD update
        launch_sgd(d_w, d_b, d_w_grad, d_b_grad, n_w, n_b);
        cudaDeviceSynchronize();
    }

    printf("Avg forward pass:  %.3f us\n", total_fwd_ms / ITERATIONS * 1000.0f);
    printf("Avg backward pass: %.3f us\n", total_bwd_ms / ITERATIONS * 1000.0f);

    cudaMemcpy(h_op, d_op[LAYERS-1], size_op[LAYERS-1], cudaMemcpyDeviceToHost);

    // Free the allocated space
    cudaEventDestroy(fwd_start); cudaEventDestroy(fwd_end);
    cudaEventDestroy(bwd_start); cudaEventDestroy(bwd_end);
    for(int s = 0; s < STREAMS; s++) cudaStreamDestroy(streams[s]);

    free(h_ip); free(h_y); free(h_op);
    for(int i = 0; i < LAYERS; i++){ free(h_w[i]); free(h_b[i]); }

    cudaFree(d_ip); cudaFree(d_y);
    for(int i = 0; i < LAYERS; i++){
        cudaFree(d_w[i]); cudaFree(d_b[i]); cudaFree(d_op[i]);
        cudaFree(d_w_grad[i]); cudaFree(d_b_grad[i]); cudaFree(d_op_grad[i]);
    }

    return 0;
}

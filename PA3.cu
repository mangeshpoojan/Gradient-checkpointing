#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define K 32
#define M 32
#define N 32  // Number of columns in B and C
#define BLOCK_SIZE 32
#define INPUT_VEC 32 
#define BATCH_SIZE 32
#define OUTPUT_VEC 32
#define STREAMS 4
#define LAYERS 4
#define LEARNING_RATE 0.001f
 
__global__ void matmul_gpu_shared_mem_tiling(float *A, float *B, float *C, float *bias, int m, int k, int n){

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

        // tileA[localRow] [localCol] = A[globalRow] [tileCol];
        // tileB[localRow] [localCol] = B[tileRow]  [globalCol];
        
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

// __global__ void add_bias(float *A, float *B, int n, int m){

//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     if(col < m){
//         for (int i = 0; i < n; i++){
//             A[i * m + col] += B[col];
//         }
//     }
// }

__global__ void relu_inplace(float *A, int size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && A[idx] < 0.0f){
        A[idx] = 0.0f;
    }
}

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

void launch_mlp_chunk(float *d_ip_chunk, float **d_w, float **d_b, float **d_op_chunk,
                      int batch_size, cudaStream_t stream){

    dim3 matmul_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 hidden_grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 output_grid((OUTPUT_VEC + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 relu_block(BLOCK_SIZE);
    dim3 relu_grid((batch_size * N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_gpu_shared_mem_tiling<<<hidden_grid, matmul_block, 0, stream>>>(
        d_ip_chunk, d_w[0], d_op_chunk[0], d_b[0], batch_size, INPUT_VEC, N);
    relu_inplace<<<relu_grid, relu_block, 0, stream>>>(d_op_chunk[0], batch_size * N);

    for(int i = 1; i < LAYERS - 1; i++){
        matmul_gpu_shared_mem_tiling<<<hidden_grid, matmul_block, 0, stream>>>(
            d_op_chunk[i - 1], d_w[i], d_op_chunk[i], d_b[i], batch_size, N, N);
        relu_inplace<<<relu_grid, relu_block, 0, stream>>>(d_op_chunk[i], batch_size * N);
    }

    matmul_gpu_shared_mem_tiling<<<output_grid, matmul_block, 0, stream>>>(
        d_op_chunk[LAYERS - 2], d_w[LAYERS - 1], d_op_chunk[LAYERS - 1],
        d_b[LAYERS - 1], batch_size, N, OUTPUT_VEC);
}

int main() {

    float *h_ip;
    float *h_w[LAYERS];
    float *h_b[LAYERS];
    float *h_op;

    float *d_ip;
    float *d_w[LAYERS];
    float *d_b[LAYERS];
    float *d_op[LAYERS]; 

    int size_w[LAYERS];
    int size_b[LAYERS];
    int size_op[LAYERS];

    int size_ip = BATCH_SIZE * INPUT_VEC * sizeof(float);

    size_w[0] = INPUT_VEC * N * sizeof(float);
    size_b[0] = N * sizeof(float);
    size_op[0] = BATCH_SIZE * N * sizeof(float);

    for(int i = 0; i < LAYERS-2; i++){
        size_w[1+i] = N * N * sizeof(float);
        size_b[1+i] = N * sizeof(float);
        size_op[1+i] = BATCH_SIZE * N * sizeof(float);
    }
    
    size_w[LAYERS - 1] = N * OUTPUT_VEC * sizeof(float);
    size_b[LAYERS - 1] = OUTPUT_VEC * sizeof(float);
    size_op[LAYERS - 1] = BATCH_SIZE * OUTPUT_VEC * sizeof(float);
    

    if(BATCH_SIZE % STREAMS != 0){
        printf("Batch size is not divisible by number of streams");
        return 0;
    }

    int batch_size = BATCH_SIZE/STREAMS;

    h_ip = (float*)malloc(size_ip);
    for (int i = 0; i < LAYERS; i++){
        h_w[i] = (float*)malloc(size_w[i]);
        h_b[i] = (float*)malloc(size_b[i]);
    }
    h_op = (float*)malloc(size_op[LAYERS-1]);

    srand(32);
    init_matrix(h_ip, BATCH_SIZE, INPUT_VEC);
    init_matrix(h_w[0], INPUT_VEC, N);
    init_matrix(h_b[0], 1, N);
    for(int i = 0; i < LAYERS-2; i++){
        init_matrix(h_w[i+1], N, N);
        init_matrix(h_b[i+1], 1, N);
    }
    init_matrix(h_w[LAYERS - 1], N, OUTPUT_VEC);
    init_matrix(h_b[LAYERS - 1], 1, OUTPUT_VEC);

    cudaMalloc(&d_ip, size_ip);
    for(int i = 0; i < LAYERS; i++ ){
        cudaMalloc(&d_w[i], size_w[i]);
        cudaMalloc(&d_b[i], size_b[i]);
        cudaMalloc(&d_op[i], size_op[i]);
    }

    cudaMemcpy(d_ip, h_ip, size_ip, cudaMemcpyHostToDevice);
    for(int i = 0; i < LAYERS; i++){
        cudaMemcpy(d_w[i], h_w[i], size_w[i], cudaMemcpyHostToDevice);
        cudaMemcpy(d_b[i], h_b[i], size_b[i], cudaMemcpyHostToDevice);
        cudaMemset(d_op[i], 0, size_op[i]);
    }

    // create the streams
    cudaStream_t streams[STREAMS];
    for(int stream_idx = 0; stream_idx < STREAMS; stream_idx++){
        cudaStreamCreate(&streams[stream_idx]);
    }

    double start_time = get_time();

    // launch the streams
    for(int stream_idx = 0; stream_idx < STREAMS; stream_idx++){
        int batch_offset = stream_idx * batch_size;
        float *d_ip_chunk = d_ip + batch_offset * INPUT_VEC; 
        float *d_op_chunk[LAYERS];
        for(int i = 0; i < LAYERS - 1; i++) {
            d_op_chunk[i] = d_op[i] + batch_offset * N;
        }
        d_op_chunk[LAYERS - 1] = d_op[LAYERS - 1] + batch_offset * OUTPUT_VEC;
        
        launch_mlp_chunk(d_ip_chunk, d_w, d_b, d_op_chunk, batch_size, streams[stream_idx]);
    
    }
    
    for(int stream_idx = 0; stream_idx < STREAMS; stream_idx++){
        cudaStreamSynchronize(streams[stream_idx]);
    }

    double end_time = get_time();

    for(int stream_idx = 0; stream_idx < STREAMS; stream_idx++){
        cudaStreamDestroy(streams[stream_idx]);
    }

    cudaMemcpy(h_op, d_op[LAYERS - 1], size_op[LAYERS - 1], cudaMemcpyDeviceToHost);

    double elapsed_time = end_time - start_time;
    printf("Time: %f microseconds\n\n", (elapsed_time * 1e6f));



    free(h_ip);
    for(int i = 0; i < LAYERS; i++){
        free(h_w[i]);
        free(h_b[i]);
    }
    free(h_op);

    cudaFree(d_ip);
    for(int i = 0; i < LAYERS; i++){
        cudaFree(d_w[i]);
        cudaFree(d_b[i]);
        cudaFree(d_op[i]);
    }

    return 0;
}

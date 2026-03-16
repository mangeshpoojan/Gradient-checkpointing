%%cuda -c "--gpu-architecture $gpu_arch"
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
 
__global__ void matmul_gpu_shared_mem_tiling(float *A, float *B, float *C, int m, int k, int n){

    __shared__ float tileA [BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float tileB [BLOCK_SIZE * BLOCK_SIZE];

    int localRow = threadIdx.y; 
    int localCol = threadIdx.x;  
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (k + BLOCK_SIZE - 1)/BLOCK_SIZE;

    

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
        C[globalRow * n + globalCol] = sum;
    }

}

__global__ void add_bias(float *A, float *B, int n, int m){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < m){
        for (int i = 0; i < n; i++){
            A[i * m + col] += B[col];
        }
    }
}

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

void launch_mlp_chunk(float *d_ip_chunk, float *d_w1,float *d_b1,float *d_op1_chunk, float *d_w2, float *d_b2, float *d_op2_chunk, int batch_size, cudaStream_t stream){

    //layer1
    dim3 blockDim1_1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim1_1((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (batch_size + BLOCK_SIZE - 1)/BLOCK_SIZE);
    
    matmul_gpu_shared_mem_tiling<<<gridDim1_1, blockDim1_1,0,stream>>>(d_ip_chunk, d_w1, d_op1_chunk, batch_size ,INPUT_VEC, N);  

    dim3 blockDim1_2(BLOCK_SIZE);
    dim3 gridDim1_2((N + BLOCK_SIZE - 1)/BLOCK_SIZE);

    add_bias<<<gridDim1_2, blockDim1_2, 0, stream>>>(d_op1_chunk, d_b1, batch_size, N);

    dim3 blockDim1_3(BLOCK_SIZE);
    dim3 gridDim1_3((batch_size * N + BLOCK_SIZE - 1)/BLOCK_SIZE);

    relu_inplace<<<gridDim1_3, blockDim1_3, 0, stream>>>(d_op1_chunk, batch_size * N);

    // layer2
    dim3 blockDim2_1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim2_1((OUTPUT_VEC + BLOCK_SIZE - 1)/BLOCK_SIZE, (batch_size + BLOCK_SIZE - 1)/BLOCK_SIZE);

    matmul_gpu_shared_mem_tiling<<<gridDim2_1, blockDim2_1, 0, stream>>>(d_op1_chunk, d_w2, d_op2_chunk, batch_size, N, OUTPUT_VEC);

    dim3 blockDim2_2(BLOCK_SIZE);
    dim3 gridDim2_2((OUTPUT_VEC + BLOCK_SIZE - 1)/BLOCK_SIZE);

    add_bias<<<gridDim2_2, blockDim2_2, 0, stream>>>(d_op2_chunk, d_b2, batch_size, OUTPUT_VEC);

}

int main() {

    float *h_ip, *h_w1, *h_b1, *h_w2, *h_b2, *h_op;
    float *d_ip, *d_w1, *d_b1 , *d_op1, *d_w2, *d_b2, *d_op2;




    int size_ip = BATCH_SIZE * INPUT_VEC * sizeof(float);
    int size_w1 = INPUT_VEC * N * sizeof(float);
    int size_b1 = N * sizeof(float);
    int size_op1 = BATCH_SIZE * N * sizeof(float);
    int size_w2 = N * OUTPUT_VEC * sizeof(float);
    int size_op2 = BATCH_SIZE * OUTPUT_VEC * sizeof(float);
    int size_b2 = OUTPUT_VEC * sizeof(float);

    if(BATCH_SIZE % STREAMS != 0){
        printf("Batch size is not divisible by number of streams");
        return 0;
    }

    int batch_size = BATCH_SIZE/STREAMS;

    h_ip = (float*)malloc(size_ip);
    h_w1 = (float*)malloc(size_w1);
    h_b1 = (float*)malloc(size_b1);
    h_w2 = (float*)malloc(size_w2);
    h_b2 = (float*)malloc(size_b2);
    h_op = (float*)malloc(size_op2);

    srand(32);
    init_matrix(h_ip, BATCH_SIZE, INPUT_VEC);
    init_matrix(h_w1, INPUT_VEC, N);
    init_matrix(h_b1, 1, N);
    init_matrix(h_w2, N, OUTPUT_VEC);
    init_matrix(h_b2, 1, OUTPUT_VEC);

    cudaMalloc(&d_ip, size_ip);
    cudaMalloc(&d_w1, size_w1);
    cudaMalloc(&d_b1, size_b1);
    cudaMalloc(&d_op1, size_op1);
    cudaMalloc(&d_w2, size_w2);
    cudaMalloc(&d_b2, size_b2);
    cudaMalloc(&d_op2, size_op2);

    cudaMemcpy(d_ip, h_ip, size_ip, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, h_w1, size_w1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, size_b1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2, size_w2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, size_b2, cudaMemcpyHostToDevice);
    cudaMemset(d_op1, 0, size_op1);
    cudaMemset(d_op2, 0, size_op2);

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
        float *d_op1_chunk = d_op1 + batch_offset * N;
        float *d_op2_chunk = d_op2 + batch_offset * OUTPUT_VEC;
    
        launch_mlp_chunk(d_ip_chunk, d_w1, d_b1, d_op1_chunk, d_w2, d_b2, d_op2_chunk, batch_size, streams[stream_idx]);
    
    }
    
    for(int stream_idx = 0; stream_idx < STREAMS; stream_idx++){
        cudaStreamSynchronize(streams[stream_idx]);
    }

    double end_time = get_time();

    for(int stream_idx = 0; stream_idx < STREAMS; stream_idx++){
        cudaStreamDestroy(streams[stream_idx]);
    }

    cudaMemcpy(h_op, d_op2, size_op2, cudaMemcpyDeviceToHost);

    double elasped_time = end_time - start_time;
    printf("Time: %f microseconds\n\n", (elasped_time * 1e6f));



    free(h_ip);
    free(h_w1);
    free(h_b1);
    free(h_w2);
    free(h_b2);
    free(h_op);

    cudaFree(d_ip);
    cudaFree(d_w1);
    cudaFree(d_b1);   
    cudaFree(d_op1);
    cudaFree(d_w2);
    cudaFree(d_b2);
    cudaFree(d_op2);

}

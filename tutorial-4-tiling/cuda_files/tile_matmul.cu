#include <stdio.h>
#include <cuda_runtime.h>

#define TILE 16

// defining the kernel
__global__ void tiledMatMul(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE][TILE];     // shared memory for the threads in a block
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;      // the actual row, col of the output matrix C where the result is to be stored
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Ceiling division ensures we process all tiles even if N is not divisible by TILE
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {

        // Boundary check for Matrix A
        if (row < N && (t * TILE + threadIdx.x) < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];      // this is flattened matrix A, thus we do the offset calculation N*row + t*TILE
        } 
        else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f; // Padding with 0 to handle the out of bounds lookups when N is not divisible by TILE
        }

        // Boundary check for Matrix B
        if ((t * TILE + threadIdx.y) < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        } 
        else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f; // Padding with 0
        }

        __syncthreads();    // wait for all threads to load the elements they are supposed to load for these tiles in A and B

        // Multiply the two tiles
        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();    // wait for every thread to finish the above multiplication else we might corrupt the next step
    }

    // Write output safely within bounds
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main()
{
    int N = 64;

    size_t size = N * N * sizeof(float);

    // Host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++){
        h_A[i] = 2.0f;
        h_B[i] = 2.0f;
    }

    // Device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy A and B to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 2D grid of 16x16 thread blocks
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Launch kernel
    tiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[N-1][N-1] = %f\n", h_C[N*N - 1]);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
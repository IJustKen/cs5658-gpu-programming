#include <stdio.h>
#include <cuda_runtime.h>

// Global VRAM variables accessible by all blocks
__device__ int global_balance = 0;       // Shared bank account
__device__ int finished_block_count = 0; // Finished blocks counter

__global__ void bankDepositKernel(int numBlocks) {

    // Every thread in every block deposits Rs. 10 into the shared account
    // Since this is atomic, the balance is safely updated
    atomicAdd(&global_balance, 10);

    // Local barrier, in each block, all the threads wait for the remaining threads to reach here
    // Let us call it Step 1
    __syncthreads();


    // I am assuming the leader thread of a block is thread 0
    // So this if statement is entered when the current block has finished step 1, that is, all the threads in this block 
    // have updated the global balance successfully

    if (threadIdx.x == 0) {
        atomicAdd(&finished_block_count, 1);    // Thus, increment the finished blocks count
        printf("[BLOCK %d] Step 1 Finished, Waiting for Others..\n\n", blockIdx.x);
        // This empty loops keeps the thread 0 of every finished block stuck here; UNTIL all blocks are done with Step 1
        while (atomicAdd(&finished_block_count, 0) < numBlocks) {
            // Waiting lol
        }
        
        // the code below failed because apparently the GPU compiler NVCC optimizes this such that the value of finished_block_count
        // might get stored in some cache, and does not get updated in each iteration.
        // This leads to a deadlock

        // while (finished_block_count < numBlocks) {
        //     // Waiting lol
        // }

    }

    // Other threads in each block will obviously skip the if statement, so we need to stop them
    // This __syncthreads stops the other threads and will now wait for their leader thread 0 to reach this point
    // that is, wait for it to exit the loop, that is wait for all blocks to finish
    // thus this point is truly the Global barrier for all the threads in all blocks
    __syncthreads();


    // Every block checks the final balance.
    if (threadIdx.x == 0) { 
        // Only printing when it is the leader thread to not make the output cluttered of course
        printf("[VERIFICATION] Block %d reads total balance: Rs.%d\n", blockIdx.x, global_balance);
    }
}

int main() {
    int numBlocks = 4;
    int threadsPerBlock = 64; // 64*4 = 256 total threads which means Rs.2,560 total expected finally

    printf("Depositing money across %d blocks..\n\n", numBlocks);

    bankDepositKernel<<<numBlocks, threadsPerBlock>>>(numBlocks);
    cudaDeviceSynchronize();    // pause CPU execution until GPU finishes executing its thing, else program will get over

    return 0;
}
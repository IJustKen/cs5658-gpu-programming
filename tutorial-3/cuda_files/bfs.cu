#include <stdio.h>
#include <limits.h>

#define N 5
#define INF INT_MAX

// CUDA kernel
__global__ void BFS(int graph[N][N], int dist[N], int level, int *changed){
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= N)     // no need to use the threads whose node ID exceeds number of nodes
        return;

    // Look at only the nodes at the current level
    if (dist[u] == level){

        // Check all the neighbors
        for (int v = 0; v < N; v++){

            // There is an edge u -> v
            if (graph[u][v] == 1){

                // If v has not been visited
                if (dist[v] > level + 1){

                    dist[v] = level + 1;

                    // new vertex was visited, this flag is to signal that we need to 
                    // search again with the next level
                    atomicExch(changed, 1);
                }
            }
        }
    }
}

int main(){
    int h_graph[N][N] =             // using adjacency matrix 
    {
        {0, 1, 1, 0, 0},
        {0, 0, 0, 1, 1},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    int h_dist[N];  // store distances to all N nodes

    // Initialize distances to infinity
    for (int i = 0; i < N; i++){
        h_dist[i] = INF;
    }
    // Source vertex distance = 0
    h_dist[0] = 0;

    // Initializing device variables
    int (*d_graph)[N];  // stores graph
    int *d_dist;    // stores distances final
    int *d_changed;     // stores the flag used to see if BFS complete or not

    // Allocating GPU memory for the above variables
    cudaMalloc(&d_graph, sizeof(h_graph));
    cudaMalloc(&d_dist, sizeof(h_dist));
    cudaMalloc(&d_changed, sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_graph, h_graph,
               sizeof(h_graph),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_dist, h_dist,
               sizeof(h_dist),
               cudaMemcpyHostToDevice);

    int level = 0;

    int threads = N;

    while (1){
        int changed = 0;

        // d_changed = 0
        cudaMemcpy(d_changed, &changed,
                   sizeof(int),
                   cudaMemcpyHostToDevice);

        // Run BFS on the current level
        BFS<<<1, threads>>>(d_graph, d_dist, level, d_changed);

        // Wait for GPU
        cudaDeviceSynchronize();

        // Copy d_changed back to CPU to read it
        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

        // No more vertices discovered, then stop the process
        if (changed == 0){
            break;
        }
        // otherwise repeat the process for the next level
        level++;
    }

    // Copy distances from GPU to CPU
    cudaMemcpy(h_dist, d_dist, sizeof(h_dist), cudaMemcpyDeviceToHost);

    // Print result
    printf("BFS distances (shortest distances) from Vertex 0:\n");

    for (int i = 0; i < N; i++){
        if (h_dist[i] == INF){
            printf("Vertex %d : INF\n", i);
        }
        else{
            printf("Vertex %d : %d\n", i, h_dist[i]);
        }
    }

    // Free GPU memory
    cudaFree(d_graph);
    cudaFree(d_dist);
    cudaFree(d_changed);

    return 0;
}
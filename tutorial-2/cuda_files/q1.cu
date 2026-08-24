#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define INF 1000000000


void buildCSR(FILE *f, int v, int e, int *col_idx, int *row_ptr, int *weights){

    int i = 0;  // Iterator
    int r = 0;  // row_ptr iterator

    int curr_s = 0; // Current source value initialized
    int prev_s = 0; // Previous source value init

    row_ptr[0] = 0;

    while(i<e){

        // ASSUMPTION: the .txt file is ascendingly ordered by the source vertex in each line (0 indexed)
        // Destination vertices could be jumbled it is ok, but source vertices will be in order for sure
        
        fscanf(f, "%d   %d  %d", &curr_s, &col_idx[i], &weights[i]);

        if(curr_s != prev_s){
            
            // while loop in case say some vertices have 0 outgoing edges, the file will just skip those 'sources'
            // increase r until it becomes equal to curr_s value

            while(r < curr_s){
                r++;
                row_ptr[r] = i;
            }

        }

        prev_s = curr_s;    // Update prev_s
        i++;    // Go next edge
    }
    
    // In case, edges are less than total vertices, fill the remaining values in row_ptr
    // this means rest vertices have no outgoing edges
    while (r < v) {
        r++;
        row_ptr[r] = e;
    }
}




void ssspCPU(int v, int source, int *col_idx, int *row_ptr, int *weights){

    if(source >= v){
        printf("INVALID SOURCE VALUE\n");
        return;
    }

    int *dist = (int*)malloc(sizeof(int) * v);
    int *visited = (int*)malloc(sizeof(int) * v);

    // Initialize distances and visited array
    for(int i = 0; i < v; i++){
        dist[i] = INF;
        visited[i] = 0;
    }

    dist[source] = 0;


    // Dijkstra's algorithm
    for(int count = 0; count < v; count++){

        // Find the unvisited vertex with the smallest distance
        int u = -1;
        int min_dist = INF;

        for(int i = 0; i < v; i++){

            if(visited[i] == 0 && dist[i] < min_dist){
                min_dist = dist[i];
                u = i;
            }

        }

        // No more reachable vertices
        if(u == -1){
            break;
        }
        // Mark vertex as visited
        visited[u] = 1;


        // Relax all edges going out from u
        for(int i = row_ptr[u]; i < row_ptr[u + 1]; i++){

            int neighbor = col_idx[i];
            int weight = weights[i];

            if(dist[u] + weight < dist[neighbor]){
                dist[neighbor] = dist[u] + weight;
            }
        }
    }


    // Print shortest distances
    printf("\nShortest distances from vertex %d:\n", source);

    for(int i = 0; i < v; i++){

        if(dist[i] == INF)
            printf("%d -> %d = INF\n", source, i);
        else
            printf("%d -> %d = %d\n", source, i, dist[i]);
    }


    free(dist);
    free(visited);
}


__global__ void ssspKernel(int v, int source, int *row_ptr, int *col_idx, int *weights, int *dist, int *visited){

    int tid = threadIdx.x;

    // Shared variables used by all threads in the block
    __shared__ int min_dist;
    __shared__ int min_vertex;

    // Initialize dist and visited
    for(int i = tid; i < v; i += blockDim.x){
        dist[i] = INF;
        visited[i] = 0;
    }

    // Make sure every thread has finished initialization
    __syncthreads();

    // Source has distance 0
    if(tid == 0){
        dist[source] = 0;
    }

    __syncthreads();


    // Dijkstra

    for(int count = 0; count < v; count++){

        // Reset shared minimum
        if(tid == 0){
            min_dist = INF;
            min_vertex = -1;
        }

        __syncthreads();

        // Find minimum-distance unvisited vertex

        for(int i = tid; i < v; i += blockDim.x){
            if(visited[i] == 0 && dist[i] < min_dist){
                atomicMin(&min_dist, dist[i]);
            }
        }

        __syncthreads();

        // At this point min_dist is correct, but we still
        // need to find WHICH vertex has that distance.

        if(tid == 0){
            for(int i = 0; i < v; i++){
                if(visited[i] == 0 && dist[i] == min_dist){
                    min_vertex = i;
                    break;
                }
            }
        }

        __syncthreads();


        // No more reachable vertices
        if(min_vertex == -1)
            break;


        // Mark selected vertex as visited
        if(tid == 0){
            visited[min_vertex] = 1;
        }

        __syncthreads();

        // Relax all edges leaving min_vertex

        int start = row_ptr[min_vertex];
        int end = row_ptr[min_vertex + 1];

        for(int i = start + tid; i < end; i += blockDim.x){

            int neighbor = col_idx[i];
            int weight = weights[i];

            int new_dist = dist[min_vertex] + weight;

            atomicMin(&dist[neighbor], new_dist);
        }

        __syncthreads();
    }
}


void ssspGPU(int v, int source, int *d_row_ptr, int *d_col_idx, int *d_weights){
    
    int *d_dist;
    int *d_visited;

    // Allocate distance and visited arrays on GPU
    cudaMalloc(&d_dist, sizeof(int) * v);
    cudaMalloc(&d_visited, sizeof(int) * v);

    // Initialize GPU arrays
    // We will do this inside the kernel
    // because d_dist and d_visited are GPU memory.

    // Launch GPU SSSP kernel
    // int threads = 32;
    // int blocks = (v + threads - 1) / threads;
    int threads = 4;
    int blocks = 1;

    ssspKernel<<<blocks, threads>>>(
        v, 
        source, 
        d_row_ptr, 
        d_col_idx,
        d_weights,
        d_dist,
        d_visited
    );

    cudaDeviceSynchronize();    // pause CPU execution until GPU finishes


    // Copy distances back to CPU
    int *dist = (int*)malloc(sizeof(int) * v);

    cudaMemcpy(dist, d_dist, sizeof(int) * v, cudaMemcpyDeviceToHost);

    // Print result
    printf("\nGPU shortest distances from vertex %d:\n", source);

    for(int i = 0; i < v; i++){

        if(dist[i] == INF)
            printf("%d -> %d = INF\n", source, i);
        else
            printf("%d -> %d = %d\n", source, i, dist[i]);
    }


    // Free GPU and CPU memory
    cudaFree(d_dist);
    cudaFree(d_visited);

    free(dist);
}


int main(){

    FILE *f = fopen("graph1.txt", "r");

    if(f == NULL){
        printf("Error opening file\n");
        return 1;
    }

    int v, e;   // my format first line has num vertices, num edges
    fscanf(f, "%d   %d", &v, &e);

    int *col_idx = (int*)malloc(sizeof(int)*e);
    int *row_ptr = (int*)malloc(sizeof(int)*(v+1));
    int *weights = (int*)malloc(sizeof(int)*e);


    buildCSR(f, v, e, col_idx, row_ptr, weights);


    printf("File opened lesgo!\n");
    printf("V = %d, E = %d\n", v, e);

    fclose(f);

    ssspCPU(v, 0, col_idx, row_ptr, weights);


    // now for the GPU

    int *d_col_idx;
    int *d_row_ptr;
    int *d_weights;

    cudaMalloc(&d_col_idx, sizeof(int) * e);
    cudaMalloc(&d_row_ptr, sizeof(int) * (v + 1));
    cudaMalloc(&d_weights, sizeof(int) * e);

    // copy the 3 arrays from CPU to GPU
    cudaMemcpy(d_col_idx, col_idx, sizeof(int) * e, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, row_ptr, sizeof(int) * (v + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, sizeof(int) * e, cudaMemcpyHostToDevice);

    ssspGPU(v, 0, d_row_ptr, d_col_idx, d_weights);

    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);
    cudaFree(d_weights);

    free(col_idx);
    free(row_ptr);
    free(weights);

    return 0;
}
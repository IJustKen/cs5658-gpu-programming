#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


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

    int INF = 1000000000;

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

    free(col_idx);
    free(row_ptr);
    free(weights);

    return 0;
}
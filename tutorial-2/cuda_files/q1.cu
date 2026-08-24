#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


int main(){

    FILE *f = fopen("graph1.txt", "r");

    if(f == NULL){
        printf("Error opening file\n");
    }

    int v, e;   // my format first line has num vertices, num edges
    fscanf(f, "%d   %d", &v, &e);

    int *col_idx = (int*)malloc(sizeof(int)*e);
    int *row_ptr = (int*)malloc(sizeof(int)*(v+1));
    int *weights = (int*)malloc(sizeof(int)*e);

    int i = 0;  // Iterator
    int r = 0;  // row_pre iterator

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


    printf("File opened lesgo!\n");
    printf("V = %d, E = %d\n", v, e);

    fclose(f);

    return 0;
}
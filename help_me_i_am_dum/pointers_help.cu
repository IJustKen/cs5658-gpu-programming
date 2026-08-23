#include<cuda.h>
#include<stdio.h>

// declare a struct, what all it holds
struct node {
        int *dist;
        float *xx;
} *head;  // additionally this also initiates a pointer variable called head, which points to a struct node (valid C syntax)
// this has been globally declared here, notice
// NOTE: dist and xx pointer variables have not been initialized with any values either


__global__ void check(int *dist, float *xx){
        printf(" check: %d %f\n",dist[0],xx[0]);
}

__global__ void check1( struct node *temp){
        int *ptr1=temp->dist;
        float *ptr2=temp->xx;
if(ptr1!=NULL &&ptr2!=NULL)     printf(" check: %d %f\n",ptr1[0],ptr2[0]);
}


int main(){
        cudaError_t err;  // hold any error from Cuda ops

        struct node temp;

        // do a malloc on GPU, ie, cudaMalloc which returns a GPU address, and put it into the pointer variable "head"
        // head -- the pointer variable containing a GPU address, living on the CPU
        // *head -- the struct node living on GPU
        if(cudaMalloc(&head, sizeof(struct node))!=cudaSuccess) printf("error allocating head");

        // Copy contents of where the head pointer points, to contents of where the temp pointer points
        // Essentially the uninitialized dist and xx pointers from the previous cudaMalloc step, 
        // are copied to the temp struct's dist and xx pointer variables
        cudaMemcpy(&temp, head, sizeof(struct node),cudaMemcpyDeviceToHost);        // Now temp struct (on CPU), contains dist and xx which point to random GPU memory
  
        // Now in temp (on CPU), intialize dist pointer variable with cudaMalloc
        // so dist points to a GPU address (but dist variable itself is on CPU)
        if(cudaMalloc(&(temp.dist),sizeof(int)*1024)!=cudaSuccess)printf("error allocating dist");

        // Same now with the xx pointer inside temp
        if(cudaMalloc(&(temp.xx),sizeof(float)*1024)!=cudaSuccess)printf("error allocating xx");

        //below statement will not work: not possible to dereference device pointer from host like head->dist
        //if(cudaMalloc(&(head->dist),sizeof(int)*1024)!=cudaSuccess)printf("error allocating dist");

//copy the temp with dist and xx fields having device address to head
        cudaMemcpy(head,&temp, sizeof(struct node),cudaMemcpyHostToDevice);
        int var=1025;
        float tt=2.5;
        
//copy one element to the zeroth position  of head->dist and head->xx on device/GPU
        cudaMemcpy(temp.dist, &var,sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(temp.xx, &tt,sizeof(float), cudaMemcpyHostToDevice);
        err=cudaGetLastError();
        check1<<<1,1>>>(head);
        cudaDeviceSynchronize();
        err=cudaGetLastError();
        if(err!=cudaSuccess)printf("%s", cudaGetErrorString(err));
        return 0;
}

# Q1 Write a program that print properties of a CUDA device (use google colab for geeting GPU). Edit program cuda.c uploaded in Moodle and print more properties.
For this, I copied the cuda.cu file given in Moodle and printed out some more properties by referring to the cuda runtime api documentation Data Structures section under cudDeviceProp.
I have separated them in the code via comments and also put an explicit print statement which states the start of the newly added properties.

# Q2 Write a program that implements barrier for all the threads of a CUDA kernel using __synchthreads and  atomic operations
For this question, I imagined a scenario where each thread in each block is trying to deposit Rs. 10 into some global bank account. 
The logic of the barrier is commented in the code itself, which uses atomicAdd(), __syncthreads() and a while loop to create the global barrier.


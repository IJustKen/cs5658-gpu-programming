## Logic
We use tiles in matrix multiplication to exploit the use of the shared memory between threads of all blocks. This reduces the number of lookups in global memory by a factor of TILE SIZE, and thus is much faster.

In my implementation I have accounted for N which is not perfectly divisible by TILE SIZE as well, in which case the out of bounds indices have been kept as = 0.
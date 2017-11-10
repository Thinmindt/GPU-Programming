/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE

    __shared__ float sdata[512];
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x; 
    int tx = threadIdx.x;

    float x = 0;
    float y = 0;
    if (i < size) {
    	x = in[i];
    }

    int secondLoad = i + blockDim.x;
    if (secondLoad < size) {
        y = in[secondLoad];
    }
    sdata[tx] = x + y;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    	if (tx < offset)
    		sdata[tx] += sdata[tx + offset];
    	__syncthreads();
    }

    if (tx == 0) {
    	out[blockIdx.x] = sdata[0];
    }
}

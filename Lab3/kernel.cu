/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

    float CVal = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    for (int i = 0; i < (TILE_SIZE + k - 1)/TILE_SIZE; i++) {
        if (i * TILE_SIZE + threadIdx.x < k && row < m)
            sharedA[threadIdx.y][threadIdx.x] = A[row*k + i*TILE_SIZE + threadIdx.x];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0.0;

        if (i * TILE_SIZE + threadIdx.y < k && col < n)
            sharedB[threadIdx.y][threadIdx.x] = B[(i*TILE_SIZE + threadIdx.y)*n+col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j)
            CVal += sharedA[threadIdx.y][j] * sharedB[j][threadIdx.x];

        __syncthreads();
    }


    if (row < m && col < n) {
        C[row * n + col] = CVal;

    }






























}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((int)ceil((float)n/BLOCK_SIZE),(int)ceil((float)m/BLOCK_SIZE));


    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

    mysgemm<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, A, B, C);


}



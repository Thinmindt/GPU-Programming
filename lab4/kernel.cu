/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
	/********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/

    //INSERT KERNEL CODE HERE

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;

	int row_i = row_o - 2;
	int col_i = col_o - 2;

	float output = 0.0f;

	__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

	if ((row_i >= 0) && (row_i < N.height) && (col_i >= 0) && (col_i < N.width))
		Ns[ty][tx] = N.elements[row_i*N.width + col_i];
	else
		Ns[ty][tx] = 0.0f;

	__syncthreads();

	if (ty < TILE_SIZE && tx < TILE_SIZE) {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				output += M_c[i][j] * Ns[i+ty][j+tx];
			}
		}

		if (row_o < P.height && col_o < P.width)
			P.elements[row_o * P.width + col_o] = output;
	}
}

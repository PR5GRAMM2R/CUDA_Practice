#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

#define BLOCK_DIM 16
#define TILE_WIDTH 16

__global__ void matrixMulKernelWIthTiling(float* M, float* N, int tileWIdth, float* result, int width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int row = by * tileWIdth + ty;
	int col = bx * tileWIdth + tx;

	float value = 0;

	for (int ph = 0; ph < ceil(width / (float)tileWIdth); ph++) {
		if ((row < width) && (ph * tileWIdth + tx) < width) {
			Mds[ty][tx] = M[row * width + ph * tileWIdth + tx];
		}
		else {
			Mds[ty][tx] = 0.0f;
		}

		if ((ph * tileWIdth + ty) < width && col < width) {
			Nds[ty][tx] = N[(ph * tileWIdth + ty) * width + col];
		}
		else {
			Nds[ty][tx] = 0.0f;
		}

		__syncthreads();

		for (int k = 0; k < tileWIdth; k++) {
			value += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}

	if (row < width && col < width) {
		result[row * width + col] = value;
	}
}

int main()
{
	int n = 512;
	int size = n * n * sizeof(float);

	float* M_h = new float[n * n];
	float* N_h = new float[n * n];
	float* result_h = new float[n * n];

	float temp = 1.0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			M_h[i * n + j] = temp + j;
			//N_h[j * n + i] = temp + j;
			N_h[i * n + j] = temp + j;
		}
		temp += 1.0;
	}

	float* M_d, * N_d, * result_d;

	cudaMalloc(&M_d, size);
	cudaMalloc(&N_d, size);
	cudaMalloc(&result_d, size);

	cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

	dim3 gridDim(ceil(n / (float)BLOCK_DIM), ceil(n / (float)BLOCK_DIM), 1);
	dim3 blockDim(BLOCK_DIM, BLOCK_DIM, 1);

	auto start = std::chrono::steady_clock::now();

	for (int repeat = 0; repeat < 10000; repeat++) {
		matrixMulKernelWIthTiling << <gridDim, blockDim >> > (M_d, N_d, TILE_WIDTH, result_d, n);
	}

	auto end = std::chrono::steady_clock::now();
	double iter_ms = std::chrono::duration<double>(end - start).count();
	// double iter_ms = std::chrono::duration<double, std::milli>(end - start).count();
	printf("Elapsed time: %f\n", iter_ms);

	cudaMemcpy(result_h, result_d, size, cudaMemcpyDeviceToHost);

	cudaFree(M_d);
	cudaFree(N_d);
	cudaFree(result_d);

	/*{
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				printf("%.1f ", result_h[i * n + j]);
			printf("\n");
		}
	}*/

	delete[] M_h;
	delete[] N_h;
	delete[] result_h;

	return 0;
}
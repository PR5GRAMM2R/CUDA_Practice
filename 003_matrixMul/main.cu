#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

#define BLOCK_DIM 16

__global__ void matrixMulKernel(float* M, float* N, float* result, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < width && col < width) {
		float value = 0;
		for (int k = 0; k < width; k++) {
			value += M[row * width + k] * N[k * width + col];
		}
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
		matrixMulKernel << <gridDim, blockDim >> > (M_d, N_d, result_d, n);
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
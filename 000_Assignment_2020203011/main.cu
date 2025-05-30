#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <stdio.h>
#include <omp.h>
#include <sstream>

using namespace std;

#define CUDA_SYNC_CHECK() cudaSyncCheck( __FILE__, __LINE__ )

inline void cudaSyncCheck(const char* file, unsigned int line)
{
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		std::stringstream ss;
		ss << "CUDA error on synchronize with error '"
			<< cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
		throw std::runtime_error(ss.str().c_str());
	}
}

#define BLOCK_DIM 32
#define TILE_WIDTH 64

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
	__syncthreads();
}


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
	__syncthreads();
}


__global__ void warming()
{
	for (int i = 0; i < 1000000; i++)
		i *= 1;
	__syncthreads();
}


/*
int matMulOpenMP(int nSize, int repeatNum)
{
	int n = nSize;
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

	double startTime = omp_get_wtime();

	for (int repeat = 0; repeat < repeatNum; repeat++) {
#pragma omp parallel for collapse(2)
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				float sum = 0;
				for (int k = 0; k < n; k++) {
					sum += M_h[i * n + k] * N_h[k * n + j];
				}
				result_h[i * n + j] = sum;
			}
		}
	}

	double endTime = omp_get_wtime();
	printf("Parallel Execution Time : %f seconds\n", endTime - startTime);

	{
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				printf("%.1f ", result_h[i * n + j]);
			printf("\n");
		}
	}

	delete[] M_h, N_h, result_h;

	return 0;
}*/


int matMulCUDA(int nSize, int repeatNum)
{
	int n = nSize;
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

	warming << <gridDim, blockDim >> > ();

	//auto start = std::chrono::steady_clock::now();

	float timeSum = 0;

	for (int repeat = 0; repeat < repeatNum; repeat++) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		matrixMulKernel << <gridDim, blockDim >> > (M_d, N_d, result_d, n);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedMs = 0.0f;
		cudaEventElapsedTime(&elapsedMs, start, stop);
		timeSum += elapsedMs;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	//auto end = std::chrono::steady_clock::now();
	//double iter_ms = std::chrono::duration<double>(end - start).count();
	// double iter_ms = std::chrono::duration<double, std::milli>(end - start).count();
	//printf("Elapsed time: %.10f\n", iter_ms / (double)repeatNum);

	printf("GPU Average Time: %f ms\n", timeSum / (double)repeatNum);

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


int matMulCUDATile(int nSize, int repeatNum)
{
	int n = nSize;
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

	warming << <gridDim, blockDim >> > ();

	//auto start = std::chrono::steady_clock::now();

	float timeSum = 0;

	for (int repeat = 0; repeat < repeatNum; repeat++) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		matrixMulKernelWIthTiling << <gridDim, blockDim >> > (M_d, N_d, TILE_WIDTH, result_d, n);
		
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedMs = 0.0f;
		cudaEventElapsedTime(&elapsedMs, start, stop);
		timeSum += elapsedMs;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	//auto end = std::chrono::steady_clock::now();
	//double iter_ms = std::chrono::duration<double>(end - start).count();
	// double iter_ms = std::chrono::duration<double, std::milli>(end - start).count();
	//printf("Elapsed time: %.10f\n", iter_ms / (double)repeatNum);

	printf("GPU Average Time: %f ms\n", timeSum / (double)repeatNum);

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


int main()
{
	//matMulOpenMP(512, 1000); => CUDA 프로젝트에서는 OpenMP 지원 안 됨.printf("=== 512 ===\n");
	/*{
		printf("=== 4 ===\n");
		matMulCUDA(4, 100);
		matMulCUDATile(4, 100);
		printf("=== 16 ===\n");
		matMulCUDA(16, 100);
		matMulCUDATile(16, 100);
		printf("=== 64 ===\n");
		matMulCUDA(64, 100);
		matMulCUDATile(64, 100);
		printf("=== 256 ===\n");
		matMulCUDA(256, 100);
		matMulCUDATile(256, 100);
		printf("=== 1024 ===\n");
		matMulCUDA(1024, 100);
		matMulCUDATile(1024, 100);
	}*/

	printf("=== 8192 ===\n");
	matMulCUDA(8192, 100);
	matMulCUDATile(8192, 100);
}
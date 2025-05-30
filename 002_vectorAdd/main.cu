#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

int main()
{
	int n = 64;
	int size = n * sizeof(float);
	float* A_h = new float[n];
	float* B_h = new float[n];
	float* C_h = new float[n];

	float* A_d, * B_d, * C_d;

	cudaMalloc(&A_d, size);
	cudaMalloc(&B_d, size);
	cudaMalloc(&C_d, size);

	cudaMemset(A_d, 0, n);
	cudaMemset(B_d, 0, n);
	cudaMemset(C_d, 0, n);

	{
		for (int i = 0; i < n; i++)
			A_h[i] = 1.0;

		for (int i = 0; i < n; i++)
			B_h[i] = 2.0;

		cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
		cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = ceil(n / (float)threadsPerBlock);

	vecAddKernel << <blocksPerGrid, threadsPerBlock >> > (A_d, B_d, C_d, n);
	
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	for (int i = 0; i < n; i++)
		printf("%.1f ", C_h[i]);

	delete[] A_h;
	delete[] B_h;
	delete[] C_h;

	return 0;
}
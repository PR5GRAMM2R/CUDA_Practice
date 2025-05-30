#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void helloCUDA(void)
{
	printf("Hello CUDA (GPU)!\n");
}

int main()
{
	printf("Hello CUDA (CPU)!\n");

	helloCUDA << <1, 10 >> > ();
	cudaDeviceSynchronize();

	return 0;
}
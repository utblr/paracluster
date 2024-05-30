#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void cuda_hello(){
  if (threadIdx.x == 0) {
    printf("Hello World from GPU!\n");
  }
}

int main() {
    cuda_hello<<<1,1>>>();
    cudaError_t cudaerr = cudaDeviceSynchronize();
    cudaCheckErrors("kernel fail");
    if (cudaerr != cudaSuccess) {
      printf("kernel launch failed with error \"%s\".\n",
	     cudaGetErrorString(cudaerr));
    }
    else {
      printf("Successful!\n");
    }

    return 0;
}

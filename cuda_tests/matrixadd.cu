#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 4

__global__ void MatAdd(int A[N*N], int B[N*N], int C[N*N]) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = row + col*N;
  printf("idx = %d\n", idx);

  if (row < N && col < N) {
    C[idx] = A[idx] + B[idx];
  }
}

int main() {
  int A[N*N];
  int B[N*N];
  int C[N*N];
  int mysize = N*N*sizeof(int);

  for (int i = 0; i < N*N; i++) {
    A[i] = i;
    B[i] = i;    
  }

  int *cudaA = 0;
  int *cudaB = 0;
  int *cudaC = 0;

  cudaMalloc(&cudaA, mysize);
  cudaMalloc(&cudaB, mysize);
  cudaMalloc(&cudaC, mysize);  

  cudaMemcpy(cudaA, A, mysize, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaB, B, mysize, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaC, C, mysize, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(N,N);

  MatAdd <<< 1, threadsPerBlock >>> (cudaA, cudaB, cudaC);
  cudaDeviceSynchronize();

  cudaMemcpy(C, cudaC, mysize, cudaMemcpyDeviceToHost);
  cudaFree(cudaC);

  for (int i = 0; i < N*N; i++) {
    printf("%d ", C[i]);
  }

  printf("\n");
  return 0;
}
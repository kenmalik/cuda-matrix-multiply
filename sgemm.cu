#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define CHECK(call) { \
    const cudaError_t cuda_ret = call; \
    if (cuda_ret != cudaSuccess) { \
        printf("Error: %s: %d, ", __FILE__, __LINE__); \
        printf("Code: %d, Reason: %s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
        exit(-1); \
    } \
}

void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h);
void matrixMulHost(int m, int k, int n, const float *A_h, const float *B_h, float* C_h);

void basicSgemm_d_1thread1element (int m, int k, int n, const float *A_h, const float *B_h, float* C_h);
void basicSgemm_d_1thread1row (int m, int k, int n, const float *A_h, const float *B_h, float* C_h);
void basicSgemm_d_1thread1column (int m, int k, int n, const float *A_h, const float *B_h, float* C_h);
__global__ void matrixMulKernel_1thread1element (int m, int k, int n, const float *A_d, const float *B_d, float* C_d);
__global__ void matrixMulKernel_1thread1row (int m, int k, int n, const float *A_d, const float *B_d, float* C_d);
__global__ void matrixMulKernel_1thread1column (int m, int k, int n, const float *A_d, const float *B_d, float* C_d);

bool isNumber(const char *string);
void fillMatrix(float *matrix, size_t rowCount, size_t colCount);
void printMatrix(const float *matrix, size_t rowCount, size_t colCount);
double myCPUTimer();
bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols);

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: sgemm [m] [k] [n]\n");
    return EXIT_FAILURE;
  }

  for (unsigned int i = 1; i <= 3; i++) {
    if (!isNumber(argv[i])) {
      fprintf(stderr, "Error: Size argument '%s' is not an integer.\n", argv[i]);
      return EXIT_FAILURE;
    }
  }

  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);

  printf("Matrix Sizes:\n");
  printf("A: %d x %d\nB: %d x %d\nC: %d x %d\n", m, k, k, n, m, n);
  printf("C total elements: %d\n\n", m * n);

  float *A = (float *) malloc(m * k * sizeof(float));
  fillMatrix(A, m, k);
  float *B = (float *) malloc(k * n * sizeof(float));
  fillMatrix(B, k, n);
  float *C = (float *) calloc(m * n, sizeof(float));

  basicSgemm_h(m, k, n, A, B, C);
  printf("\n1 Thread 1 Element\n");
  basicSgemm_d_1thread1element(m, k, n, A, B, C);
  printf("\n1 Thread 1 Row:\n");
  basicSgemm_d_1thread1row(m, k, n, A, B, C);
  printf("\n");

  free(A);
  free(B);
  free(C);

  return EXIT_SUCCESS;
}

bool isNumber(const char *string) {
  for (unsigned int i = 0; i < strlen(string); i++) {
    if (!isdigit(string[i])) {
      return false;
    }
  }
  return true;
}

void fillMatrix(float *matrix, size_t rowCount, size_t colCount) {
  for (unsigned int i = 0; i < rowCount; i++) {
    for (unsigned int j = 0; j < colCount; j++) {
      matrix[i * colCount + j] = rand() % 100 / 100.0;
    }
  }
}

void printMatrix(const float *matrix, size_t rowCount, size_t colCount) {
  for (unsigned int i = 0; i < rowCount; i++) {
    for (unsigned int j = 0; j < colCount; j++) {
      printf("%5.3f ", matrix[i * colCount + j]);
    }
    printf("\n");
  }
}

bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {
  for (unsigned int i = 0; i < nRows; i++) {
    for (unsigned int j = 0; j < nCols; j++) {
      if (abs(CPU_Answer[nCols * i + j] - GPU_Answer[nCols * i + j]) > 0.0001) {
        fprintf(stderr, "cpu: %f gpu: %f, %d %d\n", CPU_Answer[nCols * i + j], GPU_Answer[nCols * i + j], i, j);
        return false;
      }
    }
  }
  return true;
}

double myCPUTimer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
  double startTime = myCPUTimer();
  matrixMulHost(m, k, n, A_h, B_h, C_h);
  double endTime = myCPUTimer();
  printf("matrixMul on CPU: %f s\n", endTime - startTime);
}

void matrixMulHost(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int j = 0; j < n; j++) {
      float sum = 0;
      for (unsigned int element_idx = 0; element_idx < k; element_idx++) {
        sum += A_h[i * k + element_idx] * B_h[n * element_idx + j];
      }
      C_h[i * n + j] = sum;
    }
  }
}

__global__ void matrixMulKernel_1thread1element(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
  unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowIdx < m && colIdx < n) {
    float sum = 0;
    for (unsigned int i = 0; i < k; i++) {
      sum += A_d[rowIdx * k + i] * B_d[n * i + colIdx];
    }
    C_d[rowIdx * n + colIdx] = sum;
  }
}

void basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
  CHECK(cudaDeviceSynchronize());

  double startTime = myCPUTimer();

  float *A_d, *B_d, *C_d;

  double mallocStartTime = myCPUTimer();
  CHECK(cudaMalloc((void **) &A_d, m * k * sizeof(float)));
  CHECK(cudaMalloc((void **) &B_d, k * n * sizeof(float)));
  CHECK(cudaMalloc((void **) &C_d, m * n * sizeof(float)));
  CHECK(cudaDeviceSynchronize());
  double mallocEndTime = myCPUTimer();
  printf("%-68s%f s\n", "    cudaMalloc:", mallocEndTime - mallocStartTime);

  double memcpyStartTime = myCPUTimer();
  CHECK(cudaMemcpy((void *) A_d, (void *) A_h, m * k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy((void *) B_d, (void *) B_h, k * n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  double memcpyEndTime = myCPUTimer();
  printf("%-68s%f s\n", "    cudaMemcpy:", memcpyEndTime - memcpyStartTime);

  dim3 gridSize(ceil((float) n / 32), ceil((float) m / 32), 1), blockSize(32, 32, 1);
  double kernelStartTime = myCPUTimer();
  matrixMulKernel_1thread1element<<<gridSize, blockSize>>>(m, k, n, A_d, B_d, C_d);
  CHECK(cudaDeviceSynchronize());
  double kernelEndTime = myCPUTimer();
  printf("    matrixMulKernel_1thread1element<<<(%d, %d, %d), (%d, %d, %d)>>>:    %f s\n",
    gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, kernelEndTime - kernelStartTime);

  memcpyStartTime = myCPUTimer();
  CHECK(cudaMemcpy((void *) C_h, (void *) C_d, m * n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  memcpyEndTime = myCPUTimer();
  printf("%-68s%f s\n", "    cudaMemcpy:", memcpyEndTime - memcpyStartTime);

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));

  CHECK(cudaDeviceSynchronize());
  double endTime = myCPUTimer();

  printf("%-68s%f s\n\n", "matrixMultiply on GPU", endTime - startTime);

  printf("Verifying results...");
  float *cpuRes = (float *) calloc(m * n, sizeof(float));
  matrixMulHost(m, k, n, A_h, B_h, cpuRes);
  printf("%s\n", verify(cpuRes, C_h, m, n) ? "TEST PASSED" : "TEST FAILED");
  free(cpuRes);
}

__global__ void matrixMulKernel_1thread1row(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
  unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;

  if (rowIdx < m) {
    for (unsigned int colIdx = 0; colIdx < n; colIdx++) {
      float sum = 0;
      for (unsigned int i = 0; i < k; i++) {
        sum += A_d[rowIdx * k + i] * B_d[n * i + colIdx];
      }
      C_d[rowIdx * n + colIdx] = sum;
    }
  }
}

void basicSgemm_d_1thread1row(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
  CHECK(cudaDeviceSynchronize());

  double startTime = myCPUTimer();

  float *A_d, *B_d, *C_d;

  double mallocStartTime = myCPUTimer();
  CHECK(cudaMalloc((void **) &A_d, m * k * sizeof(float)));
  CHECK(cudaMalloc((void **) &B_d, k * n * sizeof(float)));
  CHECK(cudaMalloc((void **) &C_d, m * n * sizeof(float)));
  CHECK(cudaDeviceSynchronize());
  double mallocEndTime = myCPUTimer();
  printf("%-68s%f s\n", "    cudaMalloc:", mallocEndTime - mallocStartTime);

  double memcpyStartTime = myCPUTimer();
  CHECK(cudaMemcpy((void *) A_d, (void *) A_h, m * k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy((void *) B_d, (void *) B_h, k * n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  double memcpyEndTime = myCPUTimer();
  printf("%-68s%f s\n", "    cudaMemcpy:", memcpyEndTime - memcpyStartTime);

  dim3 gridSize(1, ceil((float) m / 32), 1), blockSize(1, 32, 1);
  double kernelStartTime = myCPUTimer();
  matrixMulKernel_1thread1row<<<gridSize, blockSize>>>(m, k, n, A_d, B_d, C_d);
  CHECK(cudaDeviceSynchronize());
  double kernelEndTime = myCPUTimer();
  printf("    matrixMulKernel_1thread1row<<<(%d, %d, %d), (%d, %d, %d)>>>:    %f s\n",
    gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, kernelEndTime - kernelStartTime);

  memcpyStartTime = myCPUTimer();
  CHECK(cudaMemcpy((void *) C_h, (void *) C_d, m * n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  memcpyEndTime = myCPUTimer();
  printf("%-68s%f s\n", "    cudaMemcpy:", memcpyEndTime - memcpyStartTime);

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));

  CHECK(cudaDeviceSynchronize());
  double endTime = myCPUTimer();

  printf("%-68s%f s\n\n", "matrixMultiply on GPU", endTime - startTime);

  printf("Verifying results...");
  float *cpuRes = (float *) calloc(m * n, sizeof(float));
  matrixMulHost(m, k, n, A_h, B_h, cpuRes);
  printf("%s\n", verify(cpuRes, C_h, m, n) ? "TEST PASSED" : "TEST FAILED");
  free(cpuRes);
}
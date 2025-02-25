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

bool isNumber(const char *string);
void fillMatrix(float *matrix, size_t rowCount, size_t colCount);
void printMatrix(const float *matrix, size_t rowCount, size_t colCount);
double myCPUTimer();

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
  printf("A: %d x %d\nB: %d x %d\nC: %d x %d\n\n", m, k, k, n, m, n);

  float *A = (float *) malloc(m * k * sizeof(float));
  fillMatrix(A, m, k);
  float *B = (float *) malloc(k * n * sizeof(float));
  fillMatrix(B, k, n);
  float *C = (float *) malloc(m * n * sizeof(float));

  basicSgemm_h(m, k, n, A, B, C);

  /*printf("A\n");*/
  /*printMatrix(A, m, k);*/
  /*printf("B\n");*/
  /*printMatrix(B, k, n);*/
  /*printf("C\n");*/
  /*printMatrix(C, m, n);*/

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
      float dotProduct = 0;
      for (unsigned int element_idx = 0; element_idx < k; element_idx++) {
        dotProduct += A_h[k * i + element_idx] * B_h[n * element_idx + j];
      }
      C_h[i * n + j] = dotProduct;
    }
  }
}

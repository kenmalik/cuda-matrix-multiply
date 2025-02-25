#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool isNumber(const char *string);
void fillMatrix(float *matrix, size_t rowCount, size_t colCount);
void printMatrix(float *matrix, size_t rowCount, size_t colCount);

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

  float *A = malloc(m * k * sizeof(float));
  fillMatrix(A, m, k);
  float *B = malloc(k * n * sizeof(float));
  fillMatrix(B, k, n);
  float *C = malloc(m * n * sizeof(float));

  printMatrix(A, m, k);
  printMatrix(B, k, n);

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

void printMatrix(float *matrix, size_t rowCount, size_t colCount) {
  for (unsigned int i = 0; i < rowCount; i++) {
    for (unsigned int j = 0; j < colCount; j++) {
      printf("%5.2f ", matrix[i * colCount + j]);
    }
    printf("\n");
  }
}

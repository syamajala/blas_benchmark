#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>
#include <time.h>


int main(int argc, char *argv[])
{
  char *matrix_file_path = NULL;
  char *factored_file_path = NULL;

  for(int i = 0; i < argc; i++)
  {
    if(strcmp(argv[i], "-i") == 0)
      matrix_file_path = argv[i+1];
    else if (strcmp(argv[i], "-o") == 0)
      factored_file_path = argv[i+1];
  }

  FILE *f = fopen(matrix_file_path, "r");
  int M = 0;
  int N = 0;
  int NZ = 0;

  size_t dim = 1024;
  char *line = NULL;
  ssize_t read = 0;
  read = getline(&line, &dim, f);
  read = getline(&line, &dim, f);

  fscanf(f, "%d %d %d\n", &M, &N, &NZ);
  printf("M: %d N: %d NZ: %d\n", M, N, NZ);
  double *mat = (double *)(calloc(M*N, sizeof(double)));

  for(int n = 0; n < NZ; n++)
  {
    int i = 0;
    int j = 0;
    double e = 0;
    fscanf(f, "%d %d %lg\n", &i, &j, &e);
    i--;
    j--;
    mat[j*M+i] = e;
  }

  char *uplo = "L";
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, *uplo, M, mat, N);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  u_int64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

  printf("Info: %d\n", info);
  printf("Time (microseconds): %lu\n", delta_us);

  FILE *fp = fopen(factored_file_path, "w");
  for(int i = 0; i < M; i++)
  {
    for(int j = 0; j < N; j++)
    {
      fprintf(fp, "%d %d %0.5g\n", i+1, j+1, mat[j*M+i]);
    }
  }

  free(mat);
  fclose(f);
  fclose(fp);
}

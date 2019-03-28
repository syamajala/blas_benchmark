#include "legion.h"
#include <cblas.h>
#include <lapacke.h>
#include <cstdio>
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  DPOTRF_TASK_ID
};

enum EntryFieldIDs {
  FID_IDX,
  FID_VAL
};

int main(int argc, char *argv[])
{
  FILE *f = fopen(argv[1], "r");
  int M = 0;
  int N = 0;
  int NZ = 0;
  int idx = 0;

  size_t dim = 1024;
  char *line = NULL;
  ssize_t read = 0;
  read = getline(&line, &dim, f);
  read = getline(&line, &dim, f);

  fscanf(f, "%d %d %d\n", &M, &N, &NZ);
  printf("M: %d N: %d NZ: %d\n", M, N, NZ);
}

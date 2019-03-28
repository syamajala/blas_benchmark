#include "legion.h"
#include <cblas.h>
#include <lapacke.h>
#include <cstdio>
#include <time.h>

using namespace Legion;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  DPOTRF_TASK_ID
};

enum MatrixFieldIDs
{
  FID_VAL
};

typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > AccessorRWdouble;

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  std::string input_file = "";
  std::string output_file = "";

  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-i"))
        input_file = std::string(command_args.argv[++i]);
      else if (!strcmp(command_args.argv[i],"-o"))
        output_file = std::string(command_args.argv[++i]);
    }
  }

  FILE *f = fopen(input_file.c_str(), "r");
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

  Legion::Rect<2> matrix_rect(Legion::Point<2>(0, 0), Legion::Point<2>(M-1, N-1));
  IndexSpace is = runtime->create_index_space(ctx, matrix_rect);
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double), FID_VAL);
  }

  LogicalRegion matrix_lr = runtime->create_logical_region(ctx, is, input_fs);

  runtime->fill_field(ctx, matrix_lr, matrix_lr, FID_VAL, 0.0);

  RegionRequirement req(matrix_lr, READ_WRITE, EXCLUSIVE, matrix_lr);
  req.add_field(FID_VAL);

  InlineLauncher matrix_launcher(req);
  PhysicalRegion matrix_region = runtime->map_region(ctx, matrix_launcher);

  AccessorRWdouble acc_mat(matrix_region, FID_VAL);

  for(int n = 0; n < NZ; n++)
  {
    int i = 0;
    int j = 0;
    double e = 0;
    fscanf(f, "%d %d %lg\n", &i, &j, &e);
    i--;
    j--;
    Legion::Point<2> p(i, j);
    acc_mat[p] = e;
  }

  RegionAccessor<AccessorType::Generic, double> acc = matrix_region.get_field_accessor(FID_VAL).typeify<double>();
  Domain dom = runtime->get_index_space_domain(ctx, is);
  LegionRuntime::Arrays::Rect<2> rect = dom.get_rect<2>();
  LegionRuntime::Arrays::Rect<2> tempBounds;
  ByteOffset offset[2];
  double *field = acc.raw_rect_ptr<2>(rect, tempBounds, offset);

  char uplo = 'L';
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo, M, field, N);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  u_int64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

  printf("Info: %d\n", info);
  printf("Time (microseconds): %lu\n", delta_us);

  FILE *fp = fopen(output_file.c_str(), "w");
  for (PointInRectIterator<2> pir(matrix_rect); pir(); pir++)
  {
    fprintf(fp, "%d %d %0.5g\n", pir->x+1, pir->y+1, acc_mat[*pir]);
  }

  fclose(f);
  fclose(fp);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  return Runtime::start(argc, argv);
}

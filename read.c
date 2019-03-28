#include <stdlib.h>
#include <string.h>
#include "read.h"

void read_matrix(FILE* file,
                 int nz,
                 legion_runtime_t runtime,
                 legion_context_t context,
                 legion_index_space_t is,
                 legion_physical_region_t pr[],
                 legion_field_id_t fld[])
{
  legion_accessor_array_1d_t idx_accessor = legion_physical_region_get_field_accessor_array_1d(pr[0], fld[0]);
  legion_accessor_array_1d_t val_accessor = legion_physical_region_get_field_accessor_array_1d(pr[1], fld[1]);
  legion_index_iterator_t it = legion_index_iterator_create(runtime, context, is);

  for(int n = 0; n < nz; n++)
  {
    int i = 0;
    int j = 0;
    double val = 0.0;
    fscanf(file, "%d %d %lg\n", &i, &j, &val);
    legion_ptr_t point = legion_index_iterator_next(it);
    legion_point_2d_t idx = {i-1, j-1};
    legion_accessor_array_1d_write(idx_accessor, point, &idx, sizeof(legion_point_2d_t));
    legion_accessor_array_1d_write(val_accessor, point, &val, sizeof(double));
  }
}

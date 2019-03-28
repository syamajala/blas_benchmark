#include <stdio.h>
#include "legion/legion_c.h"

void read_matrix(FILE* file,
                 int nz,
                 legion_runtime_t runtime,
                 legion_context_t context,
                 legion_index_space_t is,
                 legion_physical_region_t pr[],
                 legion_field_id_t fld[]);

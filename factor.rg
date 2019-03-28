import "regent"
local c = regentlib.c

terralib.includepath = terralib.includepath .. ";."

terralib.linklibrary("libread.so")
local read = terralib.includec("read.h")
local mmio = terralib.includec("mmio.h")

terralib.linklibrary("libcblas.so")
local cblas = terralib.includec("cblas.h")

terralib.linklibrary("liblapacke.so")
local lapack = terralib.includec("lapacke.h")

fspace Entry {
  idx:int2d,
  val:double
}

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end

local raw_ptr = raw_ptr_factory(double)

terra get_raw_ptr_2d(rect: rect2d,
                     pr : c.legion_physical_region_t,
                     fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra dpotrf_terra(rect: rect2d, m:int,
                   pr : c.legion_physical_region_t,
                   fld : c.legion_field_id_t)
  var rawA = get_raw_ptr_2d(rect, pr, fld)
  var uplo : rawstring = 'L'
  var start = c.legion_get_current_time_in_micros()
  var info = lapack.LAPACKE_dpotrf(cblas.CblasColMajor, @uplo, m, rawA.ptr, rawA.offset)
  var stop = c.legion_get_current_time_in_micros()
  c.printf("Info: %d\n", info)
  c.printf("Time: %lu\n", stop - start)
end

__demand(__leaf)
task dpotrf(rA : region(ispace(int2d), double))
where reads writes(rA)
do
  var rect = rA.bounds
  var size:int2d = rect.hi - rect.lo + {1, 1}
  dpotrf_terra(rect, size.x, __physical(rA)[0], __fields(rA)[0])
end

struct MMatBanner {
  M:int
  N:int
  NZ:int
  typecode:mmio.MM_typecode
}

terra read_matrix_banner(file : &c.FILE)
  var banner:MMatBanner
  var ret:int

  ret = mmio.mm_read_banner(file, &(banner.typecode))

  if ret ~= 0 then
    c.printf("Unable to read banner.\n")
    return MMatBanner{0, 0, 0}
  end

  var M : int[1]
  var N : int[1]
  var nz : int[1]
  ret = mmio.mm_read_mtx_crd_size(file, &(banner.M), &(banner.N), &(banner.NZ))

  if ret ~= 0 then
    c.printf("Unable to read matrix size.\n")
    return MMatBanner{0, 0, 0}
  end

  return banner
end

__demand(__leaf)
task write_matrix(mat      : region(ispace(int2d), double),
                  file     : regentlib.string,
                  banner   : MMatBanner)
where
  reads(mat)
do
  c.printf("saving matrix to: %s\n\n", file)
  var matrix_file = c.fopen(file, 'w')
  var nnz = 0

  for i in mat.ispace do
    if mat[i] ~= 0 then
      nnz += 1
    end
  end

  mmio.mm_write_banner(matrix_file, banner.typecode)
  mmio.mm_write_mtx_crd_size(matrix_file, banner.M, banner.N, nnz)

  for i in mat.ispace do
    var val = mat[i]
    if val ~= 0 then
      c.fprintf(matrix_file, "%d %d %0.5g\n", i.x+1, i.y+1, val)
    end
  end

  c.fclose(matrix_file)
end

task main()
  var args = c.legion_runtime_get_input_args()

  var matrix_file_path = ""
  var factored_file_path = ""

  for i = 0, args.argc do
    if c.strcmp(args.argv[i], "-i") == 0 then
      matrix_file_path = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-o") == 0 then
      factored_file_path = args.argv[i+1]
    end
  end

  var matrix_file = c.fopen(matrix_file_path, 'r')
  var banner = read_matrix_banner(matrix_file)

  c.printf("M: %d N: %d nz: %d typecode: %s\n", banner.M, banner.N, banner.NZ, banner.typecode)
  var mat = region(ispace(int2d, {banner.M, banner.N}), double)
  fill(mat, 0)

  var mat_entries = region(ispace(int1d, banner.NZ), Entry)
  read.read_matrix(matrix_file, banner.NZ, __runtime(), __context(),
                   __raw(mat_entries.ispace), __physical(mat_entries), __fields(mat_entries))

  for i in mat_entries.ispace do
    mat[mat_entries[i].idx] = mat_entries[i].val
  end

  dpotrf(mat)
  write_matrix(mat, factored_file_path, banner)
end

regentlib.start(main)

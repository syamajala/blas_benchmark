cmake_minimum_required(VERSION 3.9)

project(cholesky VERSION 0.0.1 DESCRIPTION "sparse cholesky")
include(GNUInstallDirs)

add_library(read SHARED mmio.c read.c)
set_target_properties(read PROPERTIES PUBLIC_HEADER "read.h")
target_compile_options(read PRIVATE -I $ENV{LG_RT_DIR})
install(TARGETS read
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(Legion_DIR /home/seshu/dev/legion/language/build)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()
add_executable(factor_legion factor_legion.cpp)
target_link_libraries(factor_legion Legion::Legion pthread lapacke cblas)

add_executable(factor_c factor_c.c)
target_link_libraries(factor_c lapacke cblas)

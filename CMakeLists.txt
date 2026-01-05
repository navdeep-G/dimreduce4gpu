cmake_minimum_required(VERSION 3.18)

project(dimreduce4gpu LANGUAGES CXX CUDA)

# ---- Build options ----
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Avoid baking fragile RPATHs into the produced shared library. Users should
# rely on LD_LIBRARY_PATH / system loader paths for CUDA libs.
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)

add_library(dimreduce4gpu SHARED)

target_compile_options(dimreduce4gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

target_include_directories(dimreduce4gpu PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

file(GLOB_RECURSE DIMREDUCE4GPU_SOURCES
  "${PROJECT_SOURCE_DIR}/src/*"
  "${PROJECT_SOURCE_DIR}/include/*"
)
target_sources(dimreduce4gpu PRIVATE ${DIMREDUCE4GPU_SOURCES})

if(OpenMP_CXX_FOUND)
  target_link_libraries(dimreduce4gpu PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(dimreduce4gpu PRIVATE
  CUDA::cublas
  CUDA::cusolver
  CUDA::cusparse
)

# ---- Output ----
# Place the built shared library inside the Python package so it can be loaded
# at runtime without an install step.
set_target_properties(dimreduce4gpu PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/dimreduce4gpu/lib"
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/dimreduce4gpu/lib"
  OUTPUT_NAME "dimreduce4gpu"
)

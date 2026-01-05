cmake_minimum_required(VERSION 3.18)

project(dimreduce4gpu LANGUAGES CXX CUDA)

# ---- Options ----
# Default CUDA architectures to build for (fatbin). You can override at configure time via:
#   -DDIMREDUCE4GPU_CUDA_ARCHES="70;75;80;86"
set(DIMREDUCE4GPU_CUDA_ARCHES "70;75;80;86" CACHE STRING "CUDA architectures to compile for")

# If the user hasn't set CMAKE_CUDA_ARCHITECTURES, use the default fatbin list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${DIMREDUCE4GPU_CUDA_ARCHES}")
endif()

# ---- Dependencies ----
find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)

# ---- Sources ----
file(GLOB_RECURSE DIMREDUCE4GPU_SOURCES CONFIGURE_DEPENDS
  "src/*"
  "include/*"
)

add_library(dimreduce4gpu SHARED ${DIMREDUCE4GPU_SOURCES})

target_include_directories(dimreduce4gpu PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(dimreduce4gpu PUBLIC cxx_std_14)
set_target_properties(dimreduce4gpu PROPERTIES
  CUDA_STANDARD 14
  CUDA_STANDARD_REQUIRED YES
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(dimreduce4gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(dimreduce4gpu PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(dimreduce4gpu PRIVATE
  CUDA::cublas
  CUDA::cusolver
  CUDA::cusparse
)

# ---- Output layout ----
# Put the built shared library where the Python package expects it.
set_target_properties(dimreduce4gpu PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/dimreduce4gpu/lib
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/dimreduce4gpu/lib
)

# Avoid baking build-path RPATHs into the artifact.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_RPATH "")
set(CMAKE_INSTALL_RPATH "")

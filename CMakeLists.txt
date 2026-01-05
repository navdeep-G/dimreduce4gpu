cmake_minimum_required(VERSION 3.18)
project(dimreduce4gpu LANGUAGES CXX)

option(DIMREDUCE4GPU_BUILD_CPU "Build CPU backend shared library" ON)
option(DIMREDUCE4GPU_BUILD_CUDA "Build CUDA backend shared library (requires CUDA toolkit)" ON)
set(DIMREDUCE4GPU_CUDA_ARCHES "70;75;80;86" CACHE STRING "CUDA architectures to build for")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Output shared libraries into a predictable runtime location.
#
# During normal development, the default output directory is the Python package's
# "dimreduce4gpu/lib" directory so the Python loader can find the .so/.dylib/.dll.
#
# During wheel builds, CI can override this to point into the wheel build
# directory (e.g., <build_lib>/dimreduce4gpu/lib) to avoid writing into the
# source tree.
set(
  DIMREDUCE4GPU_OUTPUT_DIR
  "${PROJECT_SOURCE_DIR}/dimreduce4gpu/lib"
  CACHE PATH
  "Output directory for dimreduce4gpu shared libraries"
)
file(MAKE_DIRECTORY "${DIMREDUCE4GPU_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${DIMREDUCE4GPU_OUTPUT_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${DIMREDUCE4GPU_OUTPUT_DIR}")

# ---------------- CPU backend ----------------
if(DIMREDUCE4GPU_BUILD_CPU)
  add_library(dimreduce4cpu SHARED
      src/cpu_backend.cpp
  )
  target_compile_definitions(dimreduce4cpu PRIVATE DIMREDUCE4CPU_EXPORTS=1)
  target_include_directories(dimreduce4cpu PRIVATE ${PROJECT_SOURCE_DIR}/include)

  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
    target_link_libraries(dimreduce4cpu PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

  set_target_properties(dimreduce4cpu PROPERTIES OUTPUT_NAME "dimreduce4cpu")
endif()

# ---------------- CUDA backend ----------------
if(DIMREDUCE4GPU_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(WARNING "CUDA toolkit not found; skipping CUDA backend build.")
  else()
        add_library(dimreduce4gpu SHARED
      src/matrix.cu
      src/pca.cu
      src/tsvd.cu
      src/matrix.cuh
      src/tsvd.cuh
      src/utils.cuh
      src/device_context.cuh
      include/dimreduce4gpu.h
    )
    set_target_properties(dimreduce4gpu PROPERTIES CUDA_ARCHITECTURES "${DIMREDUCE4GPU_CUDA_ARCHES}")
    target_compile_features(dimreduce4gpu PUBLIC cxx_std_17)
    target_compile_options(dimreduce4gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    target_include_directories(dimreduce4gpu PRIVATE ${PROJECT_SOURCE_DIR}/include)

    target_link_libraries(dimreduce4gpu PRIVATE CUDA::cublas CUDA::cusolver CUDA::cusparse)
    set_target_properties(dimreduce4gpu PROPERTIES OUTPUT_NAME "dimreduce4gpu")
  endif()
endif()

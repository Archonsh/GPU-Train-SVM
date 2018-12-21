cmake_minimum_required (VERSION 2.8)
project (GRID)

find_package(CUDA QUIET REQUIRED)

# add GPU-SVM library
add_library(GPU_SVM STATIC IMPORTED)
set_target_properties(GPU_SVM PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/libgpusvm.a)

# use GPU-SVM library to build the following executables: example, example2
set(PROGRAMS example example2)

foreach(program ${PROGRAMS})
	cuda_add_executable(${program} ${program}.cu)
	target_link_libraries(${program} GPU_SVM blas cuda)
	CUDA_ADD_CUBLAS_TO_TARGET(${program})
endforeach(program)

# add Grid library
cuda_add_library(Grid STATIC grid.cu multi_grid.cu ${CMAKE_BINARY_DIR}/common/setupData.cu ${CMAKE_BINARY_DIR}/common/transposeData.cu)
target_link_libraries(Grid GPU_SVM blas cuda)
cuda_add_cublas_to_target(Grid)

# use Grid library to build executable: Main
add_executable(Main main.cpp)
target_link_libraries (Main Grid)


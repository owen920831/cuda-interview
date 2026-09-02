cmake_minimum_required(VERSION 3.24)

project(cuda_kernel_course LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(course_options INTERFACE)
target_include_directories(course_options INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(course_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--expt-relaxed-constexpr>
)

function(add_course_app target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE course_options CUDA::cudart)
  set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(NAME ${target}_correctness COMMAND ${target} --check-only)
endfunction()

include(CTest)

add_course_app(device_info src/00_device_info.cu)
add_course_app(vector_add src/01_vector_add.cu)
add_course_app(reduce_sum src/02_reduce.cu)
add_course_app(transpose src/03_transpose.cu)
add_course_app(avg_pool src/05_avg_pool.cu)
add_course_app(fusion src/06_fusion.cu)
add_course_app(streams src/07_streams.cu)

add_executable(gemm src/04_gemm.cu)
target_link_libraries(gemm PRIVATE course_options CUDA::cudart CUDA::cublas)
set_target_properties(gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME gemm_correctness COMMAND gemm --check-only)

option(COURSE_BUILD_LABS "Build intentionally incomplete student labs" OFF)
if(COURSE_BUILD_LABS)
  function(add_student_lab target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE course_options CUDA::cudart)
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${target}_student_check COMMAND ${target})
  endfunction()
  add_student_lab(lab_vector_add labs/vector_add_lab.cu)
  add_student_lab(lab_reduce labs/reduce_lab.cu)
  add_student_lab(lab_transpose labs/transpose_lab.cu)
  add_student_lab(lab_gemm labs/gemm_lab.cu)
  add_student_lab(lab_avg_pool labs/avg_pool_lab.cu)
endif()

cmake_minimum_required(VERSION 3.16)
project(cuda2omp LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
enable_testing()
add_test(NAME cuda2omp-regression
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_cuda2omp.py)

install(PROGRAMS cuda2omp DESTINATION bin)
install(FILES runtime/cuda2omp_runtime.hpp DESTINATION include)

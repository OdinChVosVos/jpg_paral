cmake_minimum_required(VERSION 3.15)
project(jpeg_parallel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

find_package(OpenMP REQUIRED)

add_subdirectory(pybind11)

add_library(jpeg_lib STATIC
    src/dct.cpp
    src/quantization.cpp
    src/jpeg_encoder.cpp
)

set_target_properties(jpeg_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(jpeg_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(jpeg_lib PUBLIC OpenMP::OpenMP_CXX)

pybind11_add_module(jpeg_py src/python_bindings.cpp)
target_link_libraries(jpeg_py PRIVATE jpeg_lib)

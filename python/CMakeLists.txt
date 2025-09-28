cmake_minimum_required(VERSION 3.16)

project(edge_ai_python)

find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(edge_ai_python MODULE
    edge_ai_wrapper.cpp
)

target_include_directories(edge_ai_python PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
)

target_link_libraries(edge_ai_python PRIVATE
    edge_ai
    pybind11::module
)

set_target_properties(edge_ai_python PROPERTIES
    CXX_VISIBILITY_PRESET hidden
)
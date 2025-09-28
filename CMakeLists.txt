cmake_minimum_required(VERSION 3.16)
project(EdgeAI CXX17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_options(-march=armv8-a+crc -mtune=cortex-a76)
    add_link_options(-march=armv8-a+crc -mtune=cortex-a76)
endif()

find_package(Threads REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src/include)

# EdgeAI Library
add_library(edge_ai STATIC
    src/include/model_base.cpp
    src/ml/logistic_regression.cpp
    src/ml/decision_tree.cpp
    src/ml/knn.cpp
    src/ml/cnn.cpp
    src/utils/quantization.cpp
    src/utils/model_factory.cpp
)

target_include_directories(edge_ai PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>
    $<INSTALL_INTERFACE:include>
)

# Face Lock Example
add_executable(face_lock examples/face_lock.cpp)
target_link_libraries(face_lock edge_ai -lwiringPi)

# Iris Demo Example
add_executable(iris_demo examples/iris_demo.cpp)
target_link_libraries(iris_demo edge_ai)

# Unit Tests
enable_testing()
add_executable(test_algorithms tests/test_algorithms.cpp)
target_link_libraries(test_algorithms edge_ai)
add_test(NAME AlgorithmsTest COMMAND test_algorithms)

# Python Wrapper
option(ENABLE_PYTHON_WRAPPER "Enable Python wrapper" OFF)
if(ENABLE_PYTHON_WRAPPER)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(FATAL_ERROR "pybind11 is required for Python wrapper")
    endif()

    add_subdirectory(python)
endif()

# Install targets
install(TARGETS edge_ai
    EXPORT EdgeAITargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY src/include/ DESTINATION include)
install(FILES README.md DESTINATION .)

# Examples
install(TARGETS face_lock iris_demo
    RUNTIME DESTINATION bin/examples
)

if(ENABLE_PYTHON_WRAPPER)
    install(TARGETS edge_ai_python MODULE DESTINATION lib)
endif()
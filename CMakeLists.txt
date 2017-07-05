cmake_minimum_required(VERSION 3.6)
project(brh_neural_net)

set(CMAKE_CXX_STANDARD 14)

include_directories("../cpp_supports/src")

add_subdirectory("../cpp_allocators" "${CMAKE_CURRENT_BINARY_DIR}/cpp_allocators_build")

link_libraries(brh_cpp_supports)

set(SOURCE_FILES
    src/brh/neural_net/constant/hidden_group.h
    src/brh/neural_net/constant/network.h
    src/brh/neural_net/constant/node.h
    src/brh/neural_net/dynamic/network.h
    src/brh/neural_net/dynamic/node.h
    src/brh/neural_net/net_layout/net_layout.h
    src/brh/neural_net/activation_functions.h
    src/brh/neural_net/common.h
    src/brh/neural_net/layered.cpp
    src/brh/neural_net/layered.h
    src/brh/neural_net/main.cpp)

add_executable(brh_neural_net ${SOURCE_FILES})

target_compile_options(brh_neural_net PUBLIC -O0)
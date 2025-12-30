cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_semaphore_demo)

add_executable(metal_example_semaphore_demo)
target_sources(metal_example_semaphore_demo PRIVATE semaphore_demo.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_semaphore_demo PUBLIC TT::Metalium)

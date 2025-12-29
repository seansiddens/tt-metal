add_executable(streamk)
target_sources(streamk PRIVATE streamk.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    streamk
    PRIVATE
        TT::Metalium
        Matmul::Common
)

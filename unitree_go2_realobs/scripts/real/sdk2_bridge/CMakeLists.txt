cmake_minimum_required(VERSION 3.16)
project(go2_udp_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/unitree_sdk2")
if(NOT EXISTS "${SDK_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR "Unitree SDK2 not found at: ${SDK_ROOT}")
endif()

# Reuse SDK-provided imported target.
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${SDK_ROOT}" "${CMAKE_BINARY_DIR}/unitree_sdk2_build" EXCLUDE_FROM_ALL)

add_executable(go2_udp_bridge go2_udp_bridge.cpp)
target_link_libraries(go2_udp_bridge PRIVATE unitree_sdk2)

set_target_properties(go2_udp_bridge PROPERTIES
  BUILD_RPATH "${SDK_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}"
)

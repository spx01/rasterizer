cmake_minimum_required(VERSION 3.24)
set(pname crast)
project(${pname} LANGUAGES HIP CXX C)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw3
    GIT_REPOSITORY    https://github.com/glfw/glfw.git
    GIT_TAG           3.4
    FIND_PACKAGE_ARGS CONFIG
)

set(BOOST_INCLUDE_LIBRARIES core;format;scope_exit CACHE STRING "" FORCE)
set(BOOST_ENABLE_CMAKE ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    Boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG boost-1.88.0
    FIND_PACKAGE_ARGS CONFIG COMPONENTS core format scope_exit
)
FetchContent_MakeAvailable(Boost glfw3)

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE
        "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA."
    )
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()

enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 20)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

if(WIN32)
    set(ROCM_ROOT
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

if(GPU_RUNTIME STREQUAL "CUDA")
    include_directories("${ROCM_ROOT}/include")
endif()

add_executable(${pname} src/main.hip external/glad/glad.c)

if(UNIX)
    target_link_libraries(${pname} PRIVATE ${CMAKE_DL_LIBS})
endif()

set_source_files_properties(src/main.hip PROPERTIES LANGUAGE ${GPU_RUNTIME})

target_include_directories(${pname} PRIVATE external)
target_link_libraries(${pname} PRIVATE glfw )
target_include_directories(${pname} PRIVATE Boost::core Boost::format Boost::scope_exit)

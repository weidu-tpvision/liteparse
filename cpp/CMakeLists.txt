cmake_minimum_required(VERSION 3.20)
project(ocria_latin_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OCRIA_ENABLE_ARCH_FLAGS "Enable architecture-specific optimization flags" ON)

find_package(Threads REQUIRED)

# Normalize target architecture (supports Linux x86/x86_64, arm64/aarch64, arm32/armv7-a)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _PROC)
set(OCRIA_ARCH_FLAGS "")
if(OCRIA_ENABLE_ARCH_FLAGS AND NOT CMAKE_CXX_FLAGS MATCHES "-march=")
    if(_PROC MATCHES "^(x86_64|amd64|i[3-6]86|x86)$")
        set(OCRIA_ARCH_FLAGS -O3)
    elseif(_PROC MATCHES "^(aarch64|arm64)$")
        set(OCRIA_ARCH_FLAGS -O3)
    elseif(_PROC MATCHES "^(arm|armv7|armv7l|armv7-a|arm32)")
        set(OCRIA_ARCH_FLAGS -O3 -march=armv7-a -mfpu=neon -mfloat-abi=hard)
    else()
        set(OCRIA_ARCH_FLAGS -O3)
    endif()
else()
    set(OCRIA_ARCH_FLAGS -O3)
endif()

message(STATUS "ocria_latin_cpp target processor: ${CMAKE_SYSTEM_PROCESSOR} (flags: ${OCRIA_ARCH_FLAGS})")

# Paths to ONNX Runtime headers and shared library
set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "Directory containing onnxruntime_c_api.h and onnxruntime_cxx_api.h")
set(ORT_LIB_PATH "" CACHE FILEPATH "Path to libonnxruntime shared library")

# Auto-discover host libonnxruntime.so only for native builds (never accidentally link host .so on cross-builds)
if(NOT ORT_LIB_PATH AND NOT CMAKE_CROSSCOMPILING)
    set(_ORT_CANDIDATES
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib/libonnxruntime.so"
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/models/ocria-nano/runtime/libonnxruntime.so"
    )
    foreach(_cand IN LISTS _ORT_CANDIDATES)
        if(EXISTS "${_cand}")
            set(ORT_LIB_PATH "${_cand}")
            break()
        endif()
    endforeach()
endif()

if(NOT ORT_LIB_PATH)
    find_library(ORT_LIB_PATH NAMES onnxruntime)
endif()

if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime headers not found in ${ORT_INCLUDE_DIR}. "
        "Run `python3 cpp/scripts/fetch_headers.py` or pass -DORT_INCLUDE_DIR=/path/to/include")
endif()

if(NOT ORT_LIB_PATH)
    message(FATAL_ERROR
        "libonnxruntime.so not found for ${CMAKE_SYSTEM_PROCESSOR}. "
        "Pass -DORT_LIB_PATH=/path/to/libonnxruntime.so (or for native Linux x86_64/aarch64, "
        "run `python3 cpp/scripts/fetch_headers.py --with-runtime`).")
endif()

get_filename_component(ORT_LIB_DIR "${ORT_LIB_PATH}" DIRECTORY)

add_library(ocria_latin STATIC
    src/img_ops.cpp
    src/ort_session.cpp
    src/detector.cpp
    src/recognizer.cpp
    src/ocria_latin.cpp
)

target_include_directories(ocria_latin
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${ORT_INCLUDE_DIR}
)

target_compile_options(ocria_latin PRIVATE ${OCRIA_ARCH_FLAGS} -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(ocria_latin PUBLIC "${ORT_LIB_PATH}" Threads::Threads m)

add_executable(ocria_cli src/main.cpp)
target_include_directories(ocria_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_compile_options(ocria_cli PRIVATE ${OCRIA_ARCH_FLAGS} -Wall -Wextra)
target_link_libraries(ocria_cli PRIVATE ocria_latin)

if(CMAKE_CROSSCOMPILING)
    set_target_properties(ocria_cli PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
else()
    if(EXISTS "${ORT_LIB_PATH}")
        file(CREATE_LINK "${ORT_LIB_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.so.1" SYMBOLIC)
    endif()
    set_target_properties(ocria_cli PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${ORT_LIB_DIR};$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

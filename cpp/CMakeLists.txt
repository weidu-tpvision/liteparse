cmake_minimum_required(VERSION 3.20)
project(ocria_latin_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Paths to ONNX Runtime headers and shared library (override for custom SDKs / cross-builds)
set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "Directory containing onnxruntime_c_api.h and onnxruntime_cxx_api.h")
set(ORT_LIB_PATH "" CACHE FILEPATH "Path to libonnxruntime shared library")

if(NOT ORT_LIB_PATH)
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
        "libonnxruntime.so not found. Run `python3 cpp/scripts/fetch_headers.py --with-runtime`, "
        "`lit models pull ocria-nano`, or pass -DORT_LIB_PATH=/path/to/libonnxruntime.so")
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

target_compile_options(ocria_latin PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(ocria_latin PUBLIC "${ORT_LIB_PATH}")

add_executable(ocria_cli src/main.cpp)
target_include_directories(ocria_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_compile_options(ocria_cli PRIVATE -O3 -Wall -Wextra)
target_link_libraries(ocria_cli PRIVATE ocria_latin)

# Ensure SONAME symlink libonnxruntime.so.1 exists in the build dir for RPATH lookup
if(EXISTS "${ORT_LIB_PATH}")
    file(CREATE_LINK "${ORT_LIB_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.so.1" SYMBOLIC)
endif()

set_target_properties(ocria_cli PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${ORT_LIB_DIR}"
    INSTALL_RPATH "${CMAKE_CURRENT_BINARY_DIR};${ORT_LIB_DIR}"
)

cmake_minimum_required (VERSION 3.18...4.1.2)
project (pcg-acceleration LANGUAGES C CXX)
set(NAME "pcg-acceleration")

option(PCG_ENABLE_CUDA "Build optional CUDA polynomial-multiplication baseline" OFF)
option(PCG_ENABLE_PROFILING "Enable PCG/OLE phase profiling helpers" OFF)
option(PCG_ENABLE_GPU_DPF "Build optional GPU-DPF standalone microbenchmark" OFF)
# Prefer a sibling GPU-DPF checkout next to this repo; fall back to the
# Docker image's install path. Override with -DGPU_DPF_ROOT=... if needed.
if(EXISTS "${CMAKE_SOURCE_DIR}/../GPU-DPF/dpf_gpu/dpf/dpf_hybrid.cu")
    set(_gpu_dpf_default "${CMAKE_SOURCE_DIR}/../GPU-DPF")
elseif(EXISTS "/opt/GPU-DPF/dpf_gpu/dpf/dpf_hybrid.cu")
    set(_gpu_dpf_default "/opt/GPU-DPF")
else()
    set(_gpu_dpf_default "${CMAKE_SOURCE_DIR}/../GPU-DPF")
endif()
set(GPU_DPF_ROOT "${_gpu_dpf_default}"
    CACHE PATH "Path to the GPU-DPF checkout")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/emp-base.cmake)

find_package(emp-ot REQUIRED)
include_directories(${EMP-OT_INCLUDE_DIRS})

find_package(OpenSSL REQUIRED)

add_library(pcg_options INTERFACE)
target_include_directories(pcg_options INTERFACE ${CMAKE_SOURCE_DIR})
if(PCG_ENABLE_PROFILING)
    target_compile_definitions(pcg_options INTERFACE PCG_ENABLE_PROFILING=1)
endif()

if(PCG_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 5000 Ada = compute 8.9
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(GPUNTT CONFIG REQUIRED)
    add_library(pcg_cuda STATIC
        common/poly_mul_cuda.cu              # v1 hand-rolled baseline
        common/poly_mul_gpuntt.cu            # GPU-NTT "merge" backend (default GPU)
        common/poly_mul_gpuntt_4step.cu      # GPU-NTT "4-step" backend (PIM vehicle)
        common/aes_gpu.cu                    # device AES-128 core (DPF + LPN PRG)
        common/dpf_gpu.cu                    # GPU full-domain half-tree DPF eval
        common/leaf_convert_cuda.cu)         # device leaf->FFp fold (PCG-OLE)
    target_include_directories(pcg_cuda PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(pcg_cuda PUBLIC CUDA::cudart pcg_options GPUNTT::ntt)
    target_compile_features(pcg_cuda PUBLIC cxx_std_17)
    set_target_properties(pcg_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        # Device-link the relocatable device code into the static library itself
        # so plain C++ consumers (half_tree_dpf and, transitively, the tests)
        # can link it without performing their own CUDA device-link step.
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 17
        POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(pcg_cuda PUBLIC PCG_ENABLE_CUDA=1)
endif()

# ── half_tree_dpf library ─────────────────────────────────────────────────
add_library(half_tree_dpf STATIC half_tree_dpf/dpf.cpp)
target_include_directories(half_tree_dpf PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(half_tree_dpf PUBLIC bool_circuit pcg_options ${EMP-OT_LIBRARIES} OpenSSL::Crypto OpenSSL::SSL)
# Optional GPU DPF full-domain evaluation (common/dpf_gpu.cu in pcg_cuda).
# Linking pcg_cuda also propagates its PUBLIC PCG_ENABLE_CUDA=1 define, which
# activates the GPU dispatch path inside dpf.cpp.
if(TARGET pcg_cuda)
    target_link_libraries(half_tree_dpf PUBLIC pcg_cuda)
endif()

# ── bool_circuit library ──────────────────────────────────────────────────
add_library(bool_circuit STATIC bool_circuit/circuit.cpp)
target_include_directories(bool_circuit PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(bool_circuit PUBLIC pcg_options ${EMP-OT_LIBRARIES})

# ── pcg_ole_2pc (header-only template, depends on half_tree_dpf + bool_circuit)
# No library target — the template is instantiated in the test/application.

# ── pcg_ole (optional; needs Intel HEXL) ──────────────────────────────────
find_package(hexl QUIET)
if(hexl_FOUND)
    add_executable(pcg_ole pcg_ole.cpp)
    target_link_libraries(pcg_ole pcg_options ${EMP-OT_LIBRARIES} Intel::hexl)
else()
    message(STATUS "Intel HEXL not found — skipping pcg_ole target. "
                   "Install from https://github.com/intel/hexl to enable.")
endif()

# ── tests ────────────────────────────────────────────────────────────────
ENABLE_TESTING()
ADD_SUBDIRECTORY(test)

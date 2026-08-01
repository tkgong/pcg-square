# Testing macros - build test_<name> from <name>.cpp.

function(pcg_prepare_test_runtime _name _out_var)
    set(_dir "${CMAKE_BINARY_DIR}/test_work/${_name}")
    file(MAKE_DIRECTORY "${_dir}/data")
    set(${_out_var} "${_dir}" PARENT_SCOPE)
endfunction()

function(pcg_link_optional_cuda _target)
    if(TARGET pcg_cuda)
        target_link_libraries(${_target} pcg_cuda)
    endif()
endfunction()

macro(add_test_executable_with_lib _name _lib)
    add_executable(test_${_name} "${_name}.cpp")
    target_link_libraries(test_${_name} ${_lib} pcg_options ${EMP-OT_LIBRARIES} OpenSSL::Crypto OpenSSL::SSL)
    pcg_link_optional_cuda(test_${_name})
endmacro()

macro(add_test_case_with_run _name _lib)
    add_test_executable_with_lib(${_name} "${_lib}")
    pcg_prepare_test_runtime(${_name} _workdir)
    add_test(NAME ${_name}
             COMMAND "${CMAKE_SOURCE_DIR}/run" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}"
             WORKING_DIRECTORY "${_workdir}")
endmacro()

function(add_2pc_ctest _name _port)
    pcg_prepare_test_runtime(${_name} _workdir)
    add_test(NAME ${_name}
             COMMAND "${CMAKE_SOURCE_DIR}/run_2pc"
                     "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}"
                     "${_port}" ${ARGN}
             WORKING_DIRECTORY "${_workdir}")
endfunction()

# One-process DPF test driver.
add_test_case_with_run(half_tree_dpf half_tree_dpf)

# 2PC tests - party IDs are 1/2 (ALICE/BOB), use run_2pc script.
add_test_executable_with_lib(bool_circuit bool_circuit)
add_2pc_ctest(bool_circuit 7788)

add_test_executable_with_lib(adder bool_circuit)
add_2pc_ctest(adder 7789)

add_test_executable_with_lib(pcg_ole_2pc half_tree_dpf)
add_2pc_ctest(pcg_ole_2pc 7790 --poly-mul cpu)

add_test_executable_with_lib(triple half_tree_dpf)
add_2pc_ctest(triple 7791 --poly-mul cpu)

add_executable(bench_poly_mul bench_poly_mul.cpp)
target_link_libraries(bench_poly_mul half_tree_dpf pcg_options ${EMP-OT_LIBRARIES} OpenSSL::Crypto OpenSSL::SSL)
pcg_link_optional_cuda(bench_poly_mul)

add_executable(bench_pcg_ole_2pc bench_pcg_ole_2pc.cpp)
target_link_libraries(bench_pcg_ole_2pc half_tree_dpf pcg_options ${EMP-OT_LIBRARIES} OpenSSL::Crypto OpenSSL::SSL)
pcg_link_optional_cuda(bench_pcg_ole_2pc)

if(TARGET pcg_cuda)
    # Phase 0 gate: device AES-128 must be bit-exact vs emp-tool's AES.
    add_executable(test_aes_gpu aes_gpu.cpp)
    target_link_libraries(test_aes_gpu pcg_cuda pcg_options ${EMP-OT_LIBRARIES} OpenSSL::Crypto OpenSSL::SSL)
    pcg_prepare_test_runtime(aes_gpu _aes_workdir)
    add_test(NAME aes_gpu
             COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_aes_gpu"
             WORKING_DIRECTORY "${_aes_workdir}")
    set_tests_properties(aes_gpu PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test_poly_mul_cuda poly_mul_cuda.cpp)
    target_link_libraries(test_poly_mul_cuda pcg_cuda pcg_options)
    pcg_prepare_test_runtime(poly_mul_cuda _cuda_workdir)
    add_test(NAME poly_mul_cuda
             COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_poly_mul_cuda"
             WORKING_DIRECTORY "${_cuda_workdir}")
    set_tests_properties(poly_mul_cuda PROPERTIES SKIP_RETURN_CODE 77)

    pcg_prepare_test_runtime(pcg_ole_2pc_cuda _pcg_cuda_workdir)
    add_test(NAME pcg_ole_2pc_cuda
             COMMAND "${CMAKE_SOURCE_DIR}/run_2pc"
                     "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_pcg_ole_2pc"
                     "7792" --N 128 --t 4 --w 1 --poly-mul cuda
             WORKING_DIRECTORY "${_pcg_cuda_workdir}")
    set_tests_properties(pcg_ole_2pc_cuda PROPERTIES SKIP_RETURN_CODE 77)

    # End-to-end OLE with the GPU DPF + GPU poly-mul (skips without a device).
    pcg_prepare_test_runtime(pcg_ole_2pc_gpu_dpf _pcg_gpu_dpf_workdir)
    add_test(NAME pcg_ole_2pc_gpu_dpf
             COMMAND "${CMAKE_SOURCE_DIR}/run_2pc"
                     "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_pcg_ole_2pc"
                     "7794" --N 128 --t 4 --w 1 --poly-mul cuda --dpf gpu
             WORKING_DIRECTORY "${_pcg_gpu_dpf_workdir}")
    set_tests_properties(pcg_ole_2pc_gpu_dpf PROPERTIES SKIP_RETURN_CODE 77)

    pcg_prepare_test_runtime(triple_cuda _triple_cuda_workdir)
    add_test(NAME triple_cuda
             COMMAND "${CMAKE_SOURCE_DIR}/run_2pc"
                     "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_triple"
                     "7793" --poly-mul cuda
             WORKING_DIRECTORY "${_triple_cuda_workdir}")
    set_tests_properties(triple_cuda PROPERTIES SKIP_RETURN_CODE 77)

    if(PCG_ENABLE_GPU_DPF)
        if(EXISTS "${GPU_DPF_ROOT}/dpf_gpu/dpf/dpf_hybrid.cu")
            add_executable(bench_gpu_dpf gpu_dpf_microbench.cu)
            target_include_directories(bench_gpu_dpf PRIVATE
                "${GPU_DPF_ROOT}"
                "${GPU_DPF_ROOT}/dpf_gpu"
                "${GPU_DPF_ROOT}/dpf_gpu/prf")
            target_compile_definitions(bench_gpu_dpf PRIVATE
                GPU_DPF_ROOT="${GPU_DPF_ROOT}"
                FUSES_MATMUL=0
                MM=1)
            target_link_libraries(bench_gpu_dpf CUDA::cudart pcg_options)
            set_target_properties(bench_gpu_dpf PROPERTIES CUDA_STANDARD 17)
        else()
            message(WARNING "PCG_ENABLE_GPU_DPF=ON but GPU_DPF_ROOT is missing dpf_hybrid.cu")
        endif()
    endif()
endif()

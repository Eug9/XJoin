#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs '-device dg1'")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(project src/ops/project.dp.cpp src/ops/utils/generator.h src/ops/utils/gpu_utils.h block-functions/load.dp.hpp block-functions/store.dp.hpp block-functions/crystal.dp.hpp)
add_executable(join src/ops/join.dp.cpp src/ops/utils/generator.h src/ops/utils/gpu_utils.h block-functions/load.dp.hpp block-functions/store.dp.hpp block-functions/crystal.dp.hpp)

include_directories(./)

add_custom_target(build
  ALL
  DEPENDS project
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

add_custom_target(run
  COMMAND ../project
  DEPENDS build
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND
        MKL_SEQUENTIAL_LAYER_LIBRARY AND MKL_CORE_LIBRARY AND MKL_SYCL_LIBRARY)
  set (MKL_FIND_QUIETLY TRUE)
endif()

set(SYCL_LIB "mkl_sycl")
set(INT_LIB "mkl_intel_ilp64")
set(SEQ_LIB "mkl_sequential")
set(THR_LIB "mkl_intel_thread")
set(COR_LIB "mkl_core")

find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS $ENV{MKLROOT}/include)


set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
set(MKL_LIBRARIES mkl_sycl mkl_intel_ilp64 mkl_sequential mkl_core sycl pthread)

target_link_libraries(project ${MKL_LIBRARIES})

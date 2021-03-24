IF(CMAKE_BUILD_TYPE MATCHES "Release")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
ELSE()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
ENDIF()

add_executable(project src/ops/project.dp.cpp src/ops/utils/generator.h src/ops/utils/gpu_utils.h crystal/load.dp.hpp crystal/store.dp.hpp crystal/crystal.dp.hpp)
add_executable(join src/ops/join.dp.cpp src/ops/utils/generator.h src/ops/utils/gpu_utils.h crystal/load.dp.hpp crystal/store.dp.hpp crystal/crystal.dp.hpp)

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

#if(NOT BUILD_SHARED_LIBS)
#  set(SYCL_LIB "libmkl_sycl.a")
 # set(INT_LIB "libmkl_intel_ilp64.a")
  #set(SEQ_LIB "libmkl_sequential.a")
 # set(THR_LIB "libmkl_intel_thread.a")
  #set(COR_LIB "libmkl_core.a")
#else()
  set(SYCL_LIB "mkl_sycl")
  set(INT_LIB "mkl_intel_ilp64")
  set(SEQ_LIB "mkl_sequential")
  set(THR_LIB "mkl_intel_thread")
  set(COR_LIB "mkl_core")
#endif()

find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS $ENV{MKLROOT}/include)


set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
set(MKL_LIBRARIES mkl_sycl mkl_intel_ilp64 mkl_sequential mkl_core sycl pthread)

#if (MKL_INCLUDE_DIR AND
 #       MKL_INTERFACE_LIBRARY AND
  #      MKL_SEQUENTIAL_LAYER_LIBRARY AND
   #     MKL_CORE_LIBRARY AND MKL_SYCL_LIBRARY)

  #if (NOT DEFINED ENV{CRAY_PRGENVPGI} AND
   #       NOT DEFINED ENV{CRAY_PRGENVGNU} AND
    #      NOT DEFINED ENV{CRAY_PRGENVCRAY} AND
     #     NOT DEFINED ENV{CRAY_PRGENVINTEL})
    #set(ABI "-m64")
  #endif()

  #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMKL_ILP64 ${ABI}")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 ${ABI}")

#else()

 # set(MKL_INCLUDE_DIRS "")
  #set(MKL_LIBRARIES "")
 # set(MKL_INTERFACE_LIBRARY "")
 # set(MKL_SEQUENTIAL_LAYER_LIBRARY "")
 # set(MKL_CORE_LIBRARY "")
  #set(MKL_SYCL_LIBRARY "")
#endif()

target_link_libraries(project ${MKL_LIBRARIES})

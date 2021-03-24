#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

/*
DPCT1026:1: The call to cudaEventCreate was removed, because this call is
redundant in DPC++.
*/
#define SETUP_TIMING() sycl::event start, stop;
std::chrono::time_point<std::chrono::high_resolution_clock> start_ct1;
std::chrono::time_point<std::chrono::high_resolution_clock> stop_ct1;;;

/*
DPCT1012:2: Detected kernel execution time measurement pattern and generated an
initial code for time measurements in SYCL. You can change the way time is
measured depending on your goals.
*/
#define TIME_FUNC(f, t)                                                        \
    {                                                                          \
        start_ct1 = std::chrono::high_resolution_clock::now();                 \
        f;                                                                     \
        dpct::get_default_queue().wait_and_throw();                                                 \
        stop_ct1 = std::chrono::high_resolution_clock::now();                  \
        t =                                                         \
            std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)     \
                .count();                                                      \
    }


/**
 * ADDED BY MY
 *
 * Just For Compatibility
 * **/

void malloc_wrapper(void **ptr, size_t size){
    *ptr=malloc_device(size, dpct::get_default_queue());
}

void free_wrapper(void *vec){
    free(vec, dpct::get_default_queue());
}

#define CLEANUP(vec) if(vec) free_wrapper(vec)//CubDebugExit(g_allocator.DeviceFree(vec))

#define ALLOCATE(vec,size) (malloc_wrapper((void**)&vec, size))//CubDebugExit(g_allocator.DeviceAllocate((void**)&vec, size))

template <typename T>
T *loadToGPU(T *src, int numEntries/*,
             cub::CachingDeviceAllocator &g_allocator*/) try {
  T* dest;
  //CubDebugExit(g_allocator.DeviceAllocate((void**)&dest, sizeof(T) * numEntries));
    /*
    DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    CubDebugExit((dpct::get_default_queue()
                      .memcpy(dest, src, sizeof(T) * numEntries)
                      .wait(),
                  0));
  return dest;
}
catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

#define TILE_SIZE (BLOCK_THREADS * ITEMS_PER_THREAD)

#define CHECK_ERROR() { \
  cudaDeviceSynchronize(); \
  cudaError_t error = cudaGetLastError(); \
  if(error != cudaSuccess) \
  { \
    printf("CUDA error: %s\n", cudaGetErrorString(error)); \
    exit(-1); \
  } \
}

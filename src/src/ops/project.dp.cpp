// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdio.h>
#include "oneapi/mkl/rng.hpp"
//#include <mkl_rng_sycl.hpp>

#include <cmath>

//#include <cub/util_allocator.cuh>
//#include "cub/test/test_util.h"

#include "crystal/crystal.dp.hpp"

#include "utils/gpu_utils.h"
#include <chrono>


#include <fstream>

using namespace std;


//---------------------------------------------------------------------
// Implements Projection Operator
// There are two variants: dot-product and sigmoid
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
//cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void project(float* in1, float* in2, float* out, int num_items,
             sycl::nd_item<1> item_ct1)
{
  float items[ITEMS_PER_THREAD];
  float items2[ITEMS_PER_THREAD];
  float res[ITEMS_PER_THREAD];

  int tile_offset = item_ct1.get_group(0) * TILE_SIZE; // group_id() + 128*4
  int num_tiles = (num_items + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

    if (item_ct1.get_group(0) == num_tiles - 1) {
        num_tile_items = num_items - tile_offset; // 100 -
    }

  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in1 + tile_offset, items, num_tile_items, item_ct1);
  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in2 + tile_offset, items2, num_tile_items, item_ct1);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (item_ct1.get_local_id(0) + (ITEM * BLOCK_THREADS) <
            num_tile_items) {
      res[ITEM] = 2*items[ITEM] + 3*items2[ITEM];
    }
  }

  BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>(out + tile_offset, res, num_tile_items, item_ct1);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void projectSigmoid(float* in1, float* in2, float* out, int num_items,
                    sycl::nd_item<1> item_ct1)
{
  float items[ITEMS_PER_THREAD];
  float items2[ITEMS_PER_THREAD];
  float res[ITEMS_PER_THREAD];

    int tile_offset = item_ct1.get_group(0) * TILE_SIZE;
  int num_tiles = (num_items + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

    if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_items - tile_offset;
  }

  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in1 + tile_offset, items, num_tile_items, item_ct1);
  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in2 + tile_offset, items2, num_tile_items, item_ct1);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (item_ct1.get_local_id(0) + (ITEM * BLOCK_THREADS) <
            num_tile_items) {
            res[ITEM] =
                1.0f / (1.0f + sycl::exp(-2 * items[ITEM] - 3 * items2[ITEM]));
    }
  }

  BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>(out + tile_offset, res, num_tile_items, item_ct1);
}


float projectGPU(float* in1, float* in2, float* out, int num_items) {
  SETUP_TIMING();

  float time_proj;
  int tile_items = 128*4;
  int num_blocks = (num_items + tile_items - 1)/tile_items;
    /*
    DPCT1038:9: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    TIME_FUNC(
        ( dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            //auto dpct_global_range = f * f;

            auto f_ct0 = in1;
            auto f_ct1 = in2;
            auto f_ct2 = out;
            auto f_ct3 = num_items;

            cgh.parallel_for(
                sycl::nd_range<1>({static_cast<size_t>(num_blocks*128)},{128}),
                [=](sycl::nd_item<1> item_ct1) {
                    project<128, 4>(f_ct0, f_ct1, f_ct2, f_ct3, item_ct1);
                });
            })//.wait()
             ),
        time_proj);

  return time_proj;
}

float projectSigmoidGPU(float* in1, float* in2, float* out, int num_items) {
  SETUP_TIMING();

  float time_proj;
  int tile_items = 128*4;
  int num_blocks = (num_items + tile_items - 1)/tile_items;
    /*
    DPCT1038:10: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    TIME_FUNC(
        (dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  //          auto dpct_global_range = f * f;

            auto f_ct0 = in1;
            auto f_ct1 = in2;
            auto f_ct2 = out;
            auto f_ct3 = num_items;

            cgh.parallel_for(
                sycl::nd_range<1>({static_cast<size_t>(num_blocks*128)},{128}),
                [=](sycl::nd_item<1> item_ct1) {
                    projectSigmoid<128, 4>(f_ct0, f_ct1, f_ct2, f_ct3,
                                           item_ct1);
                });
        })//.wait()
    ),
        time_proj);

  return time_proj;
}

/**
 * Main
 */
int main(int argc, char **argv) try {
    oneapi::mkl::rng::uniform<float> distr_ct1;
    int num_items = 1 << 28;
  int num_trials          = 3;

  // Initialize command line
/*  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("n", num_items);
  args.GetCmdLineArgument("t", num_trials);
  
  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
      printf("%s "
          "[--n=<input items>] "
          "[--t=<num trials>] "
          "[--device=<device-id>] "
          "[--v] "
          "\n", argv[0]);
      exit(0);
  }*/

 

  // Initialize device
  //CubDebugExit(args.DeviceInit());

  // Allocate problem device arrays
  float *d_in1 = NULL;
  //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in1, sizeof(float) * num_items));
  d_in1 = (float*) malloc_device(num_items*sizeof(float), dpct::get_default_queue().get_device(), dpct::get_default_queue().get_context());

  float *d_in2 = NULL;
  //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in2, sizeof(float) * num_items));
  d_in2 = (float*) malloc_device(num_items*sizeof(float), dpct::get_default_queue().get_device(), dpct::get_default_queue().get_context());


  float  *d_out = NULL;
  //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * num_items));
  d_out = (float*) malloc_device(num_items*sizeof(float), dpct::get_default_queue().get_device(), dpct::get_default_queue().get_context());

  float  *d_out_sig = NULL;
  //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * num_items));
  d_out_sig = (float*) malloc_device(num_items*sizeof(float), dpct::get_default_queue().get_device(), dpct::get_default_queue().get_context());

    sycl::event start, stop;
    /*
    DPCT1026:11: The call to cudaEventCreate was removed, because this call is
    redundant in DPC++.
    */
    /*
    DPCT1026:12: The call to cudaEventCreate was removed, because this call is
    redundant in DPC++.
    */
    /**ADDED BY ME**/
    cout<<"Running on: "<<dpct::get_default_queue().get_device().get_info<sycl::info::device::name>();
    /***/

    oneapi::mkl::rng::philox4x32x10 *generator;
  int seed = 0;
    generator =
        new oneapi::mkl::rng::philox4x32x10(dpct::get_default_queue(), seed);
    /*
    DPCT1026:13: The call to curandSetPseudoRandomGeneratorSeed was removed,
    because the function call is redundant in DPC++.
    */
    oneapi::mkl::rng::generate(distr_ct1, *generator, num_items, d_in1);
    oneapi::mkl::rng::generate(distr_ct1, *generator, num_items, d_in2);

  float time_proj_gpu;
  float time_proj_sigmoid_gpu;  

  for (int t = 0; t < num_trials; t++) {
    time_proj_gpu = projectGPU(d_in1, d_in2, d_out, num_items);
    time_proj_sigmoid_gpu = projectSigmoidGPU(d_in1, d_in2, d_out_sig, num_items);

    cout<< "{"
        << "\"time_proj_gpu\":" << time_proj_gpu
        << ",\"time_proj_sigmoid_gpu\":" << time_proj_sigmoid_gpu
        << "}" << endl;
  }

/**ADDED BY ME*/
  float *buffer1 = (float*) malloc( num_items * sizeof(float) );
  float *buffer2 = (float*) malloc( num_items * sizeof(float) );

  float *buffer_sum= (float*) malloc( num_items * sizeof(float) );
  float *buffer_sig = (float*) malloc( num_items * sizeof(float) );

  dpct::get_default_queue().memcpy(buffer1, d_in1, num_items * sizeof(float) );
  dpct::get_default_queue().wait();

    /*{
        ofstream file1("d_in1");
        for (int i = 0; i < num_items; i++) {
            file1 << buffer1[i] << " ";
        }
    }*/
    dpct::get_default_queue().memcpy(buffer2, d_in2, num_items * sizeof(float) );
    dpct::get_default_queue().wait();

   /* {
        ofstream file2("d_in2");
        for (int i = 0; i < num_items; i++) {
            file2 << buffer2[i] << " ";
        }
    }*/
    dpct::get_default_queue().memcpy(buffer_sum, d_out, num_items * sizeof(float) );
    dpct::get_default_queue().wait();

    dpct::get_default_queue().memcpy(buffer_sig, d_out_sig, num_items * sizeof(float) );
    dpct::get_default_queue().wait();

   /*{
        ofstream file3("d_out_sum");
        for (int i = 0; i < num_items; i++) {
            file3 << buffer_sum[i] << std::endl;
        }
       ofstream file4("d_out_sig");
       for (int i = 0; i < num_items; i++) {
           file4 << buffer_sig[i] << std::endl;
       }
    }*/

   for(int i=0; i<num_items; i++){
       float tmp=2*buffer1[i]+3*buffer2[i];
       if(tmp!=buffer_sum[i]) {
           cerr << "AAAAAAAAAAAAAAAAAAAA!" << std::endl;
           cout<< tmp << " "<<buffer_sum[i]<<std::endl;
           exit(-1);
       }

   }

    for(int i=0; i<num_items; i++){
        float tmp=1.0f / (1.0f + sycl::exp(-2 * buffer1[i] - 3 * buffer2[i]));
        if(abs(tmp-buffer_sig[i])>0.000005) {
            cerr << "AAAAAAAAAAAAAAAAAAAA! PROBLEM SIGMOID" << std::endl;
            cout<< tmp << " "<<buffer_sig[i]<<std::endl;
            exit(-1);
        }

    }


 /****/
  // Cleanup
  if (d_in1) free(d_in1, dpct::get_default_queue()); //CubDebugExit(g_allocator.DeviceFree(d_in1));
  if (d_in2) free(d_in2, dpct::get_default_queue()); //CubDebugExit(g_allocator.DeviceFree(d_in2));
  if (d_out) free(d_out, dpct::get_default_queue()); //CubDebugExit(g_allocator.DeviceFree(d_out));
  if (d_out_sig) free(d_out_sig, dpct::get_default_queue()); //CubDebugExit(g_allocator.DeviceFree(d_out));

  return 0;
}
catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

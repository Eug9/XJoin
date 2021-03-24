// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdio.h>
#include "oneapi/mkl/rng.hpp"
//#include <mkl_rng_sycl.hpp>

//#include <cub/util_allocator.cuh>
//#include "cub/test/test_util.h"

#include "crystal/crystal.dp.hpp"

#include "utils/generator.h"
#include "utils/gpu_utils.h"
#include <chrono>

using namespace std;

/**ADDED BY ME*/
#define NUM_BLOCK_THREAD 128
#define NUM_ITEM_PER_THREAD 4
/***/


#define DEBUG 0

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void build_kernel(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots,
                  sycl::nd_item<1> item_ct1) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = item_ct1.get_group(0) * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items, item_ct1);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items, item_ct1);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, num_slots, num_tile_items, item_ct1);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void probe_kernel(int *fact_fkey, int *fact_val, int num_tuples, 
    int *hash_table, int num_slots, unsigned long long *res,
    sycl::nd_item<1> item_ct1, long long *buffer) {
  // Load a tile striped across threads
  int selection_flags[ITEMS_PER_THREAD];
  int keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
  int join_vals[ITEMS_PER_THREAD];

  unsigned long long sum = 0;

    int tile_offset = item_ct1.get_group(0) * TILE_SIZE;
  int num_tiles = (num_tuples+ TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

    if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(fact_fkey + tile_offset, keys, num_tile_items, item_ct1);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(fact_val + tile_offset, vals, num_tile_items, item_ct1);

  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, join_vals, selection_flags,
      hash_table, num_slots, num_tile_items, item_ct1);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
        if ((item_ct1.get_local_id(0) + (BLOCK_THREADS * ITEM) <
             num_tile_items))
            if (selection_flags[ITEM])
                sum += vals[ITEM] * join_vals[ITEM];//static_cast<unsigned long>(vals[ITEM]) * static_cast<unsigned long>(join_vals[ITEM]);
  }

    item_ct1.barrier(sycl::access::fence_space::local_space);

  unsigned long long aggregate = sycl::ONEAPI::reduce(item_ct1.get_group(), sum, sycl::ONEAPI::plus<>());
    //unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer, item_ct1);

    //item_ct1.barrier();

    if (item_ct1.get_local_id(0) == 0) {
        sycl::atomic<unsigned long long>(
            sycl::global_ptr<unsigned long long>(res))
            .fetch_add(aggregate);
  }
}

struct TimeKeeper {
  float time_build;
  float time_probe;
  float time_extra;
  float time_total;
};



TimeKeeper hashJoin(int *d_dim_key, int *d_dim_val, int *d_fact_fkey,
                    int *d_fact_val, int num_dim, int num_fact/*,
                    cub::CachingDeviceAllocator &g_allocator*/) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
  SETUP_TIMING();

  int* hash_table = NULL;
  unsigned long long* res;
  int num_slots = num_dim;
  float time_build, time_probe, time_memset, time_memset2;

  ALLOCATE(hash_table, sizeof(int) * 2 * num_dim);
  ALLOCATE(res, sizeof(long long));

  TIME_FUNC(q_ct1.memset(hash_table, 0, num_slots * sizeof(int) * 2).wait(),
              time_memset);

  TIME_FUNC(q_ct1.memset(res, 0, sizeof(long long)).wait(), time_memset2);


  //int tile_items = 128*4;
    int tile_items = NUM_BLOCK_THREAD*NUM_ITEM_PER_THREAD;
    /*
    DPCT1038:3: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    //cout<<"Max sub-group size: "<<q_ct1.get_device().get_info<sycl::info::device::sub_group_sizes>()<<std::endl;

    TIME_FUNC(
        (q_ct1.submit([&](sycl::handler &cgh) {
           // auto dpct_global_range = f * f;


            auto f_ct0 = d_dim_key;
            auto f_ct1 = d_dim_val;
            auto f_ct2 = num_dim;
            auto f_ct3 = hash_table;
            auto f_ct4 = num_slots;

            /***
             * ADDED BY ME
             *
             * **/
            size_t local_range_size = NUM_BLOCK_THREAD;
            size_t num_groups = static_cast<size_t>(num_dim + tile_items - 1) / tile_items;
            size_t global_range_size= local_range_size * num_groups;
            /****/

            cgh.parallel_for<class BuildKernel>(
                    sycl::nd_range<1>({global_range_size},{local_range_size}),
                [=](sycl::nd_item<1> item_ct1) {
                    build_kernel<NUM_BLOCK_THREAD, NUM_ITEM_PER_THREAD>(f_ct0, f_ct1, f_ct2, f_ct3, f_ct4,
                                         item_ct1);
                });
        })),
        time_build);
    /*
    DPCT1038:4: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    TIME_FUNC(
        (q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<long long, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                buffer_acc_ct1(sycl::range<1>(32), cgh);

            //auto dpct_global_range = f * f;

            auto f_ct0 = d_fact_fkey;
            auto f_ct1 = d_fact_val;
            auto f_ct2 = num_fact;
            auto f_ct3 = hash_table;
            auto f_ct4 = num_slots;
            auto f_ct5 = res;
            /***
                        * ADDED BY ME
                        *
                        * **/

            size_t local_range_size = NUM_BLOCK_THREAD;
            size_t num_groups = static_cast<size_t>(num_fact + tile_items - 1) / tile_items;
            size_t global_range_size = local_range_size * num_groups;
            /****/

            cgh.parallel_for<class ProbeKernel>(
                    sycl::nd_range<1>({global_range_size},{local_range_size}),
                [=](sycl::nd_item<1> item_ct1 )  {
                    probe_kernel<NUM_BLOCK_THREAD, NUM_ITEM_PER_THREAD>(f_ct0, f_ct1, f_ct2, f_ct3, f_ct4,
                                         f_ct5, item_ct1,
                                         buffer_acc_ct1.get_pointer());
                });
        })),
        time_probe);


    unsigned long long h_res;

    q_ct1.memcpy(&h_res, res, sizeof(long long)).wait();

    cout<<"JOIN RESULTS: "<<h_res<<std::endl;

#if DEBUG
  cout << "{" << "\"time_memset\":" << time_memset
      << ",\"time_build\"" << time_build
      << ",\"time_probe\":" << time_probe << "}" << endl;
#endif

  CLEANUP(hash_table);
  CLEANUP(res);

  TimeKeeper t = {time_build, time_probe, time_memset, time_build + time_probe + time_memset};
  return t;
}
catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
//cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


#define CLEANUP(vec) if(vec) free_wrapper(vec) //CubDebugExit(g_allocator.DeviceFree(vec))

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char **argv) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
  int num_fact           = 256 * 1<<20;
  int num_dim            = 16 * 1<<10;
  int num_trials         = 3;

  // Initialize command line
  if(argc>1) {
      num_dim = atoi(argv[1]);
  }
 /** CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("n", num_fact);
  args.GetCmdLineArgument("d", num_dim);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
        "[--n=<num fact>] "
        "[--d=<num dim>] "
        "[--t=<num trials>] "
        "[--device=<device-id>] "
        "[--v] "
        "\n", argv[0]);
    exit(0);
  }
*/

  cout<<"Running on "<<q_ct1.get_device().get_info<sycl::info::device::name>()<<std::endl;
  //cout<<"Max work item dimension: "<<q_ct1.get_device().get_info<sycl::info::device::max_work_item_dimensions>()<<std::endl;
  cout<<"Max work group size: "<<q_ct1.get_device().get_info<sycl::info::device::max_work_group_size>()<<std::endl;



  int log2 = 0;
  int num_dim_dup = num_dim >> 1;
  while (num_dim_dup) {
    num_dim_dup >>= 1;
    log2 += 1;
  }

  // Initialize device
  //CubDebugExit(args.DeviceInit());

  // Allocate problem device arrays
  int *d_dim_key = NULL;
  int *d_dim_val = NULL;
  int *d_fact_fkey = NULL;
  int *d_fact_val = NULL;

  /**
   *    This
   * */

    d_dim_key = (int*) malloc_device(sizeof(int) * num_dim, dev_ct1.default_queue());
    d_dim_val = (int*) malloc_device(sizeof(int) * num_dim, dev_ct1.default_queue());
    d_fact_fkey = (int*) malloc_device(sizeof(int) * num_fact, dev_ct1.default_queue());
    d_fact_val = (int*) malloc_device(sizeof(int) * num_fact, dev_ct1.default_queue());


    /*** Instead of this
     *
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dim_key, sizeof(int) * num_dim));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dim_val, sizeof(int) * num_dim));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_fkey, sizeof(int) * num_fact));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_val, sizeof(int) * num_fact));
      */
  int *h_dim_key = NULL;
  int *h_dim_val = NULL;
  int *h_fact_fkey = NULL;
  int *h_fact_val = NULL;

  create_relation_pk(h_dim_key, h_dim_val, num_dim);
  create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim);



  /***
   ADDED BY ME FOR CORRECTNESS CHECK
   */
    /*cout<<"| TABLE DIM |"<<std::endl<<std::endl;
    for(int i=0; i<num_dim; i++){
      cout<<h_dim_key[i]<<" "<<h_dim_val[i]<<std::endl;
    }
    cout<<"|================================================|"<<std::endl<<std::endl<<std::endl;
    cout<<"| TABLE FACT |"<<std::endl<<std::endl;
    for(int i=0; i<num_fact; i++){
        cout<<h_fact_fkey[i]<<" "<<h_fact_val[i]<<std::endl;
    }

    cout<<"|================================================|"<<std::endl<<std::endl<<std::endl;
    cout<<"| JOIN TABLE |"<<std::endl<<std::endl;
*/

    {
     /*   ofstream file("rest_host");
        int count_matches = 0;
        unsigned long long res_host = 0;
        for (int i = 0; i < num_dim; i++) {
            for (int j = 0; j < num_fact; j++) {
                if (h_dim_key[i] == h_fact_fkey[j]) {
                    //cout << h_fact_fkey[i] << " " <<  h_fact_fkey[i] << " " << h_fact_fkey[j] << " " << h_fact_val[j] << std::endl;
                    count_matches++;
                    unsigned long long u_val1=0, u_val2=0;
                    u_val1=h_dim_val[i];
                    u_val2=h_fact_val[j];
                    res_host +=  static_cast<unsigned long long>(h_dim_val[i]) * static_cast<unsigned long long>(h_fact_val[j]);
                    //if (count_matches < 1500)
                       // file << res_host << " : " << h_dim_val[i] << " " << h_fact_val[j] << " " << (unsigned long long)(h_dim_val[i] * h_fact_val[j]) << std::endl;
                }
            }
        }


        cout << "Num tuple join table: " << count_matches << std::endl;
        cout << "RES HOST JOIN: " << (unsigned long long) res_host << std::endl;*/
    }
  /****/


    /*
    DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    //CubDebugExit(
        (q_ct1.memcpy(d_dim_key, h_dim_key, sizeof(int) * num_dim).wait());//, 0));
    /*
    DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    //CubDebugExit(
        (q_ct1.memcpy(d_dim_val, h_dim_val, sizeof(int) * num_dim).wait());//, 0));
    /*
    DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    //CubDebugExit(
        (q_ct1.memcpy(d_fact_fkey, h_fact_fkey, sizeof(int) * num_fact).wait());//,
         //0));
    /*
    DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    //CubDebugExit(
        (q_ct1.memcpy(d_fact_val, h_fact_val, sizeof(int) * num_fact).wait());//,
        // 0));

  for (int j = 0; j < num_trials; j++) {
    TimeKeeper t = hashJoin(d_dim_key, d_dim_val, d_fact_fkey, d_fact_val, num_dim, num_fact/*, g_allocator*/);
    cout<< "{"
        << "\"num_dim\":" << num_dim
        << ",\"num_fact\":" << num_fact
        << ",\"radix\":" << 0
        << ",\"time_partition_build\":" << 0
        << ",\"time_partition_probe\":" << 0
        << ",\"time_partition_total\":" << 0
        << ",\"time_build\":" << t.time_build
        << ",\"time_probe\":" << t.time_probe
        << ",\"time_extra\":" << t.time_extra
        << ",\"time_join_total\":" << t.time_total
        << "}" << endl;
  }

  CLEANUP(d_dim_key);
  CLEANUP(d_dim_val);
  CLEANUP(d_fact_fkey);
  CLEANUP(d_fact_val);

  return 0;
}
catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__dpct_inline__ T BlockSum(T item, T *shared, sycl::nd_item<1> item_ct1) {
  item_ct1.barrier();

  /**
   * ADDED BY ME BUT NOT USED... TO CHECK CORRECTNESS
   * **/

  T val = item;

  const int warp_size = item_ct1.get_sub_group().dimensions;

  int lane = item_ct1.get_local_id(0) % warp_size;
  int wid = item_ct1.get_local_id(0) / warp_size;

    for (int offset = 16; offset > 0; offset /= 2) {
        val += item_ct1.get_sub_group().template shuffle_down(val, offset); //__shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) {
        shared[wid] = val;
    }

    item_ct1.get_sub_group().barrier();
    // Load the sums into the first warp
    val =
        (item_ct1.get_local_id(0) < item_ct1.get_local_range().get(0) / warp_size)
        ? shared[lane]
        : 0;

    // Calculate sum of sums
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += item_ct1.get_sub_group().template shuffle_down(val, offset);//__shfl_down_sync(0xffffffff, val, offset);
        }
    }


    /***/
          /*
  T val = item;
  const int warp_size = 32;
  int lane = item_ct1.get_local_id(0) % warp_size;
  int wid = item_ct1.get_local_id(0) / warp_size;

  // Calculate sum across warp
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  // Store sum in buffer
  if (lane == 0) {
    shared[wid] = val;
  }

  item_ct1.barrier();

  // Load the sums into the first warp
  val =
      (item_ct1.get_local_id(0) < item_ct1.get_local_range().get(0) / warp_size)
          ? shared[lane]
          : 0;

  // Calculate sum of sums
  if (wid == 0) {
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
  }*/

  return val;
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__dpct_inline__ T BlockSum(T (&items)[ITEMS_PER_THREAD], T *shared,
                           sycl::nd_item<1> item_ct1) {
  T thread_sum = 0;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    thread_sum += items[ITEM];
  }

  return BlockSum(thread_sum, shared, item_ct1);
}

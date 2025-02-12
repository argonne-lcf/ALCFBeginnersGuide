// https://github.com/argonne-lcf/sycltrain/blob/master/9_sycl_of_hell/3_nd_range.cpp

#include <stdlib.h>
#include <math.h>

#include <sycl/sycl.hpp>
#include "mpi.h"

#define _N 1024

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

class Kernel_VecAdd;

void _vecadd(sycl::queue Q, real_t * a, real_t * b, real_t * c, int N)
{

  Q.submit([&](sycl::handler &cgh) {

    cgh.parallel_for<Kernel_VecAdd>(N, [=](sycl::id<1> idx) {
      c[idx] = a[idx] + b[idx];
    }); // End of the kernel function

  }).wait();       // End of the queue commands

}

int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);

  MPI_Comm world = MPI_COMM_WORLD;

  int me, nranks;
  MPI_Comm_size(world, &nranks);
  MPI_Comm_rank(world, &me);

  const int N = _N;

  std::vector<real_t> a(N);
  std::vector<real_t> b(N);
  std::vector<real_t> c(N);
  for(int i=0; i<N; ++i) {
    a[i] = sin(i)*sin(i);
    b[i] = cos(i)*cos(i);
    c[i] = -1.0;	  
  }

  int num_devices = 0;
  if(me == 0) std::cout << "List Platforms and Devices" << std::endl;
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
  for (const auto &plat : platforms) {
    // get_info is a template. So we pass the type as an `arguments`.
    if(me == 0) std::cout << "Platform: "
			  << plat.get_info<sycl::info::platform::name>() << " "
			  << plat.get_info<sycl::info::platform::vendor>() << " "
			  << plat.get_info<sycl::info::platform::version>() << std::endl;
    // Trivia: how can we loop over argument?
    
    std::vector<sycl::device> devices = plat.get_devices();
    for (const auto &dev : devices) {
      if(me == 0) std::cout << "-- Device: "
			    << dev.get_info<sycl::info::device::name>() << " "
			    << (dev.is_gpu() ? "is a gpu" : " is not a gpu") << std::endl;
      num_devices++;
      // sycl::info::device::device_type exist, one can use a Map to translate to a nice sting
      // https://github.com/intel/llvm/blob/ef4e6ddce56fe21621bca7d62932713152de3f0e/sycl/source/detail/device_filter.cpp#L22
    }
  }

  int device_id = me % num_devices;

  sycl::queue Q( platforms[device_id].get_devices()[0] );

  for(int i=0; i<nranks; ++i) {
    if(i == 0 && me == 0) {
      printf("Running on %i ranks\n",nranks);
      printf("# of devices= %i\n\n",num_devices);
#ifdef _SINGLE_PRECISION
      printf("Using single-precision\n\n");
#else
      printf("Using double-precision\n\n");
#endif
      printf("Max WorkGroup Size= %zu\n", Q.get_device().get_info<sycl::info::device::max_work_group_size>());
    }
    if(i == me) printf("Rank %i running on GPU %i!\n",me,device_id);
    MPI_Barrier(world);
  }

  real_t * d_a = sycl::malloc_device<real_t>(N, Q);
  real_t * d_b = sycl::malloc_device<real_t>(N, Q);
  real_t * d_c = sycl::malloc_device<real_t>(N, Q);
  
  Q.memcpy(d_a, a.data(), N*sizeof(real_t));
  Q.memcpy(d_b, b.data(), N*sizeof(real_t));
  Q.memcpy(d_c, c.data(), N*sizeof(real_t));

  Q.wait();

  _vecadd(Q, d_a, d_b, d_c, N);

  Q.memcpy(c.data(), d_c, N*sizeof(real_t));
  Q.wait();

  double diff = 0.0;
  for(int i=0; i<N; ++i) diff += c[i] * c[i];
  diff = sqrt(diff)/(double) N - 1.0;

  double sum;
  MPI_Reduce(&diff, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
  if(me == 0) {
    if(sum < 1e-6) printf("\nResult is CORRECT!! :)\n");
    else printf("\nResult is WRONG!! :(\n");
  }

  free(d_a, Q);
  free(d_b, Q);
  free(d_c, Q);

  a.clear();
  b.clear();
  c.clear();

  MPI_Finalize();
}


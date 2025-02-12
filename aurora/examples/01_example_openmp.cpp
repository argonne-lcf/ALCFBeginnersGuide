#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>
#include "mpi.h"

#include <omp.h>

#define _N 1024

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

// ----------------------------------------------------------------

void _vecadd(real_t * a, real_t * b, real_t * c, int N)
{

#pragma omp target teams distribute parallel for 
  for(int i=0; i<N; ++i) {
    c[i] = a[i] + b[i];
  }

}

// ----------------------------------------------------------------
						 
int main( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);

  int me,nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  const int N = _N;

  const int bytes = N * sizeof(real_t);

  real_t * a = (real_t*) malloc(bytes);
  real_t * b = (real_t*) malloc(bytes);
  real_t * c = (real_t*) malloc(bytes);

  // Initialize host
  for(int i=0; i<N; ++i) {
    a[i] = sin(i)*sin(i);
    b[i] = cos(i)*cos(i);
    c[i] = -1.0;
  }

  int host = omp_get_initial_device();

  int num_devices = omp_get_num_devices();

  if(me == 0) {
    printf("# of devices= %i\n",num_devices);
  }

  // Device ID

  int device_id = me % num_devices;
  for(int i=0; i<nranks; ++i) {
    if(i == me) {
      printf("Rank %i on host %i running on GPU %i!\n",me,host,device_id);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  omp_set_default_device(device_id);

#ifdef _SINGLE_PRECISION
  if(me == 0) printf("Using single-precision\n\n");
#else
  if(me == 0) printf("Using double-precision\n\n");
#endif

  // Create device buffers and transfer data to device

  double * d_a = (double *) omp_target_alloc(bytes, device_id);
  double * d_b = (double *) omp_target_alloc(bytes, device_id);
  double * d_c = (double *) omp_target_alloc(bytes, device_id);

  omp_target_memcpy(d_a, a, bytes, 0, 0, device_id, host);
  omp_target_memcpy(d_b, b, bytes, 0, 0, device_id, host);
  omp_target_memcpy(d_c, c, bytes, 0, 0, device_id, host);

  // Execute kernel

  _vecadd(d_a, d_b, d_c, N);

  // Transfer data from device

  omp_target_memcpy(c, d_c, bytes, 0, 0, host, device_id);

  //Check result on host

  double diff = 0;
  for(int i=0; i<N; ++i) diff += (double) c[i];
  diff = diff/(double) N - 1.0;

  double diffsq = diff * diff;

  // Clean up

  omp_target_free(d_a, device_id);
  omp_target_free(d_b, device_id);
  omp_target_free(d_c, device_id);

  free(a);
  free(b);
  free(c);

  double sum;
  MPI_Reduce(&diffsq, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(me == 0) {
    if(sum < 1e-6) printf("\nResult is CORRECT!! :)\n");
    else printf("\nResult is WRONG!! :(\n");
  }

  MPI_Finalize();
}

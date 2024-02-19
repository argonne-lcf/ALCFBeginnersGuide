# Compilers on Polaris

This section describes how to compile C/C++ code standalone, with CUDA, and with MPI. Specifically it introduces the Cray/HPE environment for compiling system compatible codes. 

### User is assumed to know:
* how to compile and run code
* basic familiarity with CUDA and MPI
### Learning Goals:
* HPE (formerly Cray) provides custom wrappers for C/C++/FORTRAN compilers
* How to compile a C++ code
* How to compile a C++ code with CUDA and/or MPI
* Modifications to job submission script when using MPI

# Compiling C/C++ code

When you first login to Polaris, there will be a default list of loaded modules (see them with `module list`). This includes the Cray/HPE build environment. The Cray/HPE build system utilizes compiler wrappers:
- `cc` - C compiler (use it like GNU `gcc`)
- `CC` - C++ compiler (use it like GNU `g++`)
- `ftn` - Fortran compiler (use it like GNU `gfortran`)

Next an example C++ code is compiled.

### Example code: [`01_example.cpp`](examples/01_example.cpp)
```c++
#include <iostream>

int main(void){

   std::cout << "Hello World!\n";
   return 0;
}
```

Build and run on a Polaris login node or worker node
```bash
CC 01_example.cpp -o 01_example
./01_example
```

__NOTE:__ that this only uses the CPU. CUDA is required to use the GPU.

![example_cpp](media/01_compilers_cpp_example.gif)

# Compiling C/C++ with CUDA

### Example code: [`01_example.cu`](examples/01_example.cu)
```c++
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>

// create a GPU and CPU kernel to calculate the Matrix Multiplication:
// A * B = C

// GPU
__global__
void naive_matrix_multiply_gpu(double const * const A, double const * const B, double * const C, const int a, const int b, const int c)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if( row < a && col < c){
    // do the multiplication for one row and col
    double value = 0;
    for(int k = 0; k < b; k++){
      value += A[row * b + k] * B[k * c + col];
    }
    // store the result
    C[row * c + col] = value;
  }
}

// CPU
void naive_matrix_multiply_cpu(double const * const A, double const * const B, double * const C, const int a, const int b, const int c)
{

  for(int row=0;row<a;++row){
   for(int col=0;col<c;++col){
     // check boundry conditions
     if( row < a && col < c){
       // do the multiplication for one row and col
       double value = 0;
       for(int k = 0; k < b; k++){
         value += A[row * b + k] * B[k * c + col];
       }
       // store the result
       C[row * c + col] = value;
     }
   }
  }
}

int main(void){
   // set random number seed
   srand(2323);
   // matrix dimensions A = [ a x b ], B = [ b x c ], A x B = [ a x c ]
   const int a = 350;
   const int b = 400;
   const int c = 600;

   // CUDA thread block size
   const int BLOCK_SIZE = 16;
   dim3 thread_block(BLOCK_SIZE,BLOCK_SIZE,1);
   // CUDA Grid size is set by the output dimensions of A x B
   const int GRID_SIZE_ROW = ceilf(a/(float)BLOCK_SIZE);
   const int GRID_SIZE_COL = ceilf(c/(float)BLOCK_SIZE);
   dim3 grid_block(GRID_SIZE_ROW,GRID_SIZE_COL,1);

   // create host-side matrices
   double A[ a * b ] = {0.};
   double B[ b * c ] = {0.};
   double C[ a * c ] = {0.};
   // fill host-side matrices randomly
   for(int ai=0;ai<a;++ai)
      for(int bi=0;bi<b;++bi)
         A[ai*a + bi] = 1.*(double)rand()/(double)RAND_MAX;

   for(int bi=0;bi<b;++bi)
      for(int ci=0;ci<c;++ci)
         B[bi*b + ci] = 1.*(double)rand()/(double)RAND_MAX;

   // create device-side matrices, allocate device-side memory
   double *d_A, *d_B, *d_C;
   cudaMalloc(&d_A,a*b*sizeof(double));
   cudaMalloc(&d_B,b*c*sizeof(double));
   cudaMalloc(&d_C,a*c*sizeof(double));

   // copy A & B from host to device
   cudaMemcpy(d_A,A,a*b*sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(d_B,B,b*c*sizeof(double),cudaMemcpyHostToDevice);

   // Run the GPU kernel and time it.
   auto start = std::chrono::steady_clock::now();
   naive_matrix_multiply_gpu<<<grid_block,thread_block>>>(d_A,d_B,d_C,a,b,c);
   cudaDeviceSynchronize();
   auto end = std::chrono::steady_clock::now();
   std::chrono::duration<double> elapsed_seconds = end-start;
   std::cout << "gpu time: " << elapsed_seconds.count() << "\n";

   // Run the CPU kernel and time it.
   start = std::chrono::steady_clock::now();
   naive_matrix_multiply_cpu(A,B,C,a,b,c);
   end = std::chrono::steady_clock::now();
   elapsed_seconds = end-start;
   std::cout << "cpu time: " << elapsed_seconds.count() << "\n";

   return 0;
}
```

Next compile the example using on the login node:
```bash
# compile using CC, the compute version is already included
CC 01_example.cu -o 01_example_cu
```

### Submit script: [`01_example_cu.sh`](examples/01_example_cu.sh)
```bash
#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A <project-name>
#PBS -o logs/
#PBS -e logs/

/path/to/01_example_cu
```

and submit your job:
```bash
qsub -A <project-name> 01_example.sh
```

The output should look like this in the `logs/<jobID>.<hostname>.OU` file:
```
gpu time: 0.000716863
cpu time: 0.274146
```

![example_cu](media/01_compilers_cu_example.gif)


# Compiling C/C++ with MPI

Now you can compile your C++ with CUDA code, but that will only run on a single Polaris node. In order to run the same application across many nodes MPI is needed. The previous code can be augmented as follows to utilize MPI in a trivial way.

### Example Code: [`01_example_mpi.cu`](examples/01_example_mpi.cu)

```C++
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <mpi.h>

// A * B = C
__global__
void naive_matrix_multiply_gpu(double const * const A, double const * const B, double * const C, const int a, const int b, const int c)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if( row < a && col < c){
    // do the multiplication for one row and col
    double value = 0;
    for(int k = 0; k < b; k++){
      value += A[row * b + k] * B[k * c + col];
    }
    // store the result
    C[row * c + col] = value;
  }
}

void naive_matrix_multiply_cpu(double const * const A, double const * const B, double * const C, const int a, const int b, const int c)
{

  for(int row=0;row<a;++row){
   for(int col=0;col<c;++col){
     // check boundry conditions
     if( row < a && col < c){
       // do the multiplication for one row and col
       double value = 0;
       for(int k = 0; k < b; k++){
         value += A[row * b + k] * B[k * c + col];
       }
       // store the result
       C[row * c + col] = value;
     }
   }
  }
}

int main(void){

   // Initialize the MPI environment
   MPI_Init(NULL, NULL);

   // Get the number of processes
   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   // Get the rank of the process
   int world_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

   // Print off a hello world message
   printf("Hello world from rank %d out of %d processors\n",
           world_rank, world_size);

   srand(2323);
   // matrix dimensions A = [ a x b ], B = [ b x c ], A x B = [ a x c ]
   const int a = 350;
   const int b = 400;
   const int c = 600;

   // CUDA thread block size
   const int BLOCK_SIZE = 16;
   dim3 thread_block(BLOCK_SIZE,BLOCK_SIZE,1);
   // CUDA Grid size is set by the output dimensions of A x B
   const int GRID_SIZE_ROW = ceilf(a/(float)BLOCK_SIZE);
   const int GRID_SIZE_COL = ceilf(c/(float)BLOCK_SIZE);
   dim3 grid_block(GRID_SIZE_ROW,GRID_SIZE_COL,1);

   // create host-side matrices
   double A[ a * b ] = {0.};
   double B[ b * c ] = {0.};
   double C[ a * c ] = {0.};

   // fill host-side matrices randomly
   for(int ai=0;ai<a;++ai)
      for(int bi=0;bi<b;++bi)
         A[ai*a + bi] = 1.*(double)rand()/(double)RAND_MAX;

   for(int bi=0;bi<b;++bi)
      for(int ci=0;ci<c;++ci)
         B[bi*b + ci] = 1.*(double)rand()/(double)RAND_MAX;

   // create device-side matrices, allocate device-side memory
   double *d_A, *d_B, *d_C;
   cudaMalloc(&d_A,a*b*sizeof(double));
   cudaMalloc(&d_B,b*c*sizeof(double));
   cudaMalloc(&d_C,a*c*sizeof(double));

   // copy A & B from host to device
   cudaMemcpy(d_A,A,a*b*sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(d_B,B,b*c*sizeof(double),cudaMemcpyHostToDevice);

   // Run the GPU kernel and time it.
   auto start = std::chrono::steady_clock::now();
   naive_matrix_multiply_gpu<<<grid_block,thread_block>>>(d_A,d_B,d_C,a,b,c);
   cudaDeviceSynchronize();
   auto end = std::chrono::steady_clock::now();
   std::chrono::duration<double> elapsed_seconds = end-start;
   std::cout << "gpu time: " << elapsed_seconds.count() << "\n";

   // Run the CPU kernel and time it.
   start = std::chrono::steady_clock::now();
   naive_matrix_multiply_cpu(A,B,C,a,b,c);
   end = std::chrono::steady_clock::now();
   elapsed_seconds = end-start;
   std::cout << "cpu time: " << elapsed_seconds.count() << "\n";

   // Finalize the MPI environment.
   MPI_Finalize();

   return 0;
}
```

### Building the code

On Polaris, when you login there is a module for MPI already loaded, named `cray-mpich` which is a custom build of the [MPICH](https://www.mpich.org/) MPI library. It defines the location of the library installed via the environment variable `MPICH_DIR`. In order to build our code

```bash
CC 01_example_mpi.cu -o 01_example_mpi
```


### Running the code: [`01_example_mpi.sh`](examples/01_example_mpi.sh)

Next this bash script can be used to submit a 2 node job with 8 ranks, 4 per node.

```bash
#!/bin/bash
#PBS -l select=2
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A datascience
#PBS -o logs/
#PBS -e logs/

# Count number of nodes assigned
NNODES=`wc -l < $PBS_NODEFILE`
# set 1 MPI rank per GPU
NRANKS_PER_NODE=4
# calculate total ranks
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --hostfile ${PBS_NODEFILE} /home/parton/ALCFBeginnersGuide/polaris/examples/01_example_mpi

```

![example_mpi](media/01_compilers_mpi_example.gif)


# [NEXT ->](02_profiling.md)

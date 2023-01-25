#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>

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

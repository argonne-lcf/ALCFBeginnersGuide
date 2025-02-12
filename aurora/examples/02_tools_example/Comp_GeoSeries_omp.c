/*
   Description: Computing geometric series with given ratios
 
                                written by JaeHyuk Kwack (jkwack@anl.gov)
  ------------------------------------------------------------------------------ 
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if !defined(NOMPI)
#include <mpi.h>
#endif
#include <omp.h>

// Precision
#if defined(SP)
   typedef float REAL;
#else
   typedef double REAL;
#endif
int PR=sizeof(REAL);


void Comp_Geo(REAL *GeoR, REAL *GeoResult, int n, int nGeo){
   int iGeo, i, j, id;
   REAL tmpR, tmpResult;

#pragma omp target teams distribute parallel for collapse(2)
   for(j=0;j<n;j++){
      for(i=0;i<n;i++){
         id = i+j*n;
         tmpR = GeoR[id];
         tmpResult = 1.0E0;
         for (iGeo=1;iGeo<=nGeo;iGeo++){
            tmpResult = 1.0E0 + tmpR*tmpResult;
         }
         GeoResult[id] = tmpResult;
      }
   }

}


int main(int argc, char *argv[]){

   // Default array size and order
   int n=1024;
   int nGeo=30;
   
   REAL *GeoR;                   // Ratio for the series
   REAL *GeoResult;              // Result
   REAL GeoRef, GeoComp;         // For validation
   int i,j,id;

   int niter=10;
   int iiter;

   // Performance data and validation
   double Error_OMP, Error_MPI_Min, Error_MPI_Mean, Error_MPI_Max;   // Relative errors
   double FLOP_OMP, FLOP_MPI;                                        // Number of FLOPs
   double WT_OMP, WT_MPI, TIC, TOC;
   double FR_OMP, FR_MPI_Min, FR_MPI_Mean, FR_MPI_Max, FR_MPI;       // Flop-rates

   // For MPI
   int myid=0, numprocs=1;
#if !defined(NOMPI)
   MPI_Init( &argc, &argv);
   MPI_Comm_rank (MPI_COMM_WORLD, &myid);
   MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
#endif
   if (myid == 0) {
      printf("\t%40s%6d\n","Number of MPI process: ",numprocs);
   }

   // Reading command line arguments
   if (argc >=2) n = atoi(argv[1]);
   if (argc >=3) nGeo = atoi(argv[2]);

   if (myid == 0) {
      if (PR == 4) printf("\t%40s%6s\n","Precision: ","single");
      if (PR == 8) printf("\t%40s%6s\n","Precision: ","double");
      printf("\t%40s%6d\n","Number of rows/columns of the matrix: ",n);
      printf("\t%40s%6d\n","The highest order of geometric series: ",nGeo);
      printf("\t%40s%6d\n","Number of repetitions: ",niter);
      printf("\t%40s%12.6f%s\n","Memory Usage per MPI rank: ",2*1.0E-6*n*n*PR," MB");
   }

   //
   // Creating matrices
   //
   GeoR = (REAL*)malloc(n*n*PR);
   GeoResult = (REAL*)malloc(n*n*PR);

   // Initilizing GeoR
   for(j=0;j<n;j++){
      for(i=0;i<n;i++){
         GeoR[i+j*n] = 1.0E0/(2+i+j);
      }
   }

   // Computing the exact solution
   GeoRef = 0.0E0;
   for(j=0;j<n;j++){
      for(i=0;i<n;i++){
         id=i+j*n;
         GeoRef = GeoRef + (1.0E0-pow(GeoR[id],nGeo+1))/(1.0E0-GeoR[id]);
      }
   }

#pragma omp target enter data map(to:GeoR[0:n*n-1]) map(alloc:GeoResult[0:n*n-1])

   //
   // Warming up
   //
   if (myid == 0) printf("\t\tWarming up .....\n");
   Comp_Geo(GeoR, GeoResult, n, nGeo);

   //
   // Starting the main computation
   //
   TIC = omp_get_wtime();
   if(myid==0) {
      printf("\t\t%s%3d%s\n","Main Computations ",niter," repetitions ......");
      printf("\t\t%s%24s%25s%24s%25s\n\t\t","0%","25%","50%","75%","100%");
   }
   // A loop for iterations
   for(iiter=1;iiter<=niter;iiter++){
      if(myid==0) printf("|||||");
      Comp_Geo(GeoR, GeoResult, n, nGeo);
      if(myid==0) printf("|||||");
   }
   if(myid==0) printf("\n");
   TOC=omp_get_wtime();
   //
   // Ending the main computation
   //

#pragma omp target exit data map(from:GeoResult[0:n*n-1]) map(release:GeoR[0:n*n-1])

   //
   // Post-processing
   //

   // Computing Error
   GeoComp = 0.0E0;
   for(j=0;j<n;j++){
      for(i=0;i<n;i++){
         GeoComp = GeoComp + GeoResult[i+j*n];
      }
   }
   Error_OMP = fabs(GeoComp-GeoRef)/GeoRef;
   Error_MPI_Mean = 0.0E0;
#if !defined(NOMPI)
   MPI_Reduce(&Error_OMP,&Error_MPI_Min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Reduce(&Error_OMP,&Error_MPI_Max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&Error_OMP,&Error_MPI_Mean,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
#else
   Error_MPI_Min = Error_OMP;
   Error_MPI_Mean = Error_OMP;
   Error_MPI_Max = Error_OMP;
#endif
   Error_MPI_Mean = Error_MPI_Mean/numprocs;

   // Computing FLOPs
   FLOP_OMP = 1.0E0*n*n*2*nGeo*niter;
   WT_OMP = TOC-TIC;
   FLOP_MPI = 0.0E0;
   FR_MPI_Mean = 0.0E0;
   FR_OMP = FLOP_OMP/WT_OMP*1.0E-9;
#if !defined(NOMPI)
   MPI_Reduce(&WT_OMP,&WT_MPI,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&FLOP_OMP,&FLOP_MPI,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   MPI_Reduce(&FR_OMP,&FR_MPI_Min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Reduce(&FR_OMP,&FR_MPI_Max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&FR_OMP,&FR_MPI_Mean,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
#else
   WT_MPI = WT_OMP;
   FLOP_MPI = FLOP_OMP;
   FR_MPI_Min = FR_OMP;
   FR_MPI_Max = FR_OMP;
   FR_MPI_Mean = FR_OMP;
#endif
   FR_MPI = FLOP_MPI/WT_MPI*1.0E-9;
   FR_MPI_Mean = FR_MPI_Mean/numprocs;

   // Report 
   if(myid==0) {
      printf("\t\t%45s%12.4e  %12.4e  %12.4e\n","Error_MPI_{Min,Mean,Max}/MPI = ",Error_MPI_Min,Error_MPI_Mean,Error_MPI_Max);
      printf("\t\t%45s%12.6f  %12.6f  %12.6f\n","GFLOP-rate_{Min,Mean,Max}/MPI = ",FR_MPI_Min,FR_MPI_Mean,FR_MPI_Max);
      printf("\t\t%45s%12.6f%s\n","Wall Time = ",WT_MPI," sec");
      printf("\t\t%45s%12.6f%s\n","FLOP-rate = ",FR_MPI," GFLOP/sec");
   }

   // Freeing matrices
   free(GeoR);
   free(GeoResult);

#if !defined(NOMPI)
   // MPI finalize
   MPI_Finalize();
#endif
   return(0);

}


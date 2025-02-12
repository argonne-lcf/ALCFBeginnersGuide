#CC=icx
CC=mpicc

SDIR=./
SRC=${SDIR}/Comp_GeoSeries_omp.c

EXE_DP=Comp_GeoSeries_omp_${CC}_DP
EXE_DP_DEBUG=Comp_GeoSeries_omp_${CC}_DP_DEBUG
#EXE_SP=Comp_GeoSeries_omp_${CC}_SP
CFLAGS_PROFILE=-O2 -fdebug-info-for-profiling -gline-tables-only
CFLAGS_DEBUG=-g -O0
CFLAGS_COMMON=-fiopenmp -fopenmp-targets=spir64 
#CFLAGS=-fiopenmp -fopenmp-targets=spir64 -cpp
LDFLAGS=

EXES=$(EXE_DP) $(EXE_SP) $(EXE_DP_DEBUG)

all: $(EXES)

$(EXE_DP): $(SRC)
	$(CC) $(CFLAGS_COMMON) $(CFLAGS_PROFILE) $^ -o $@ $(LDFLAGS)
	rm -rf *.o *.mod *.dSYM

$(EXE_DP_DEBUG): $(SRC)
	$(CC) $(CFLAGS_COMMON) $(CFLAGS_DEBUG)  $^ -o $@ $(LDFLAGS)
	rm -rf *.o *.mod *.dSYM


$(EXE_SP): $(SRC)
	$(CC) $(CFLAGS) -DSP $^ -o $@ $(LDFLAGS)
	rm -rf *.o *.mod *.dSYM


.SUFFIXES:
.SUFFIXES: .c .o .f90 .cu .cpp .cuf
.PHONY: clean
clean:
	rm -rf *.o *.mod $(EXES) *.dSYM



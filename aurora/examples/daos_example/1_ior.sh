#!/bin/bash -x
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -A datascience
#PBS -q debug
#PBS -k doe
#PBS -ldaos=daos_user

# qsub -l select=512:ncpus=208 -l walltime=01:00:00 -A datascience -l filesystems=flare -q debug  -ldaos=daos_user  ./pbs_script.sh or - I 

export TZ='/usr/share/zoneinfo/US/Central'
date
module use /soft/modulefiles
module load daos
env | grep DRPC                                     #optional
ps -ef|grep daos                                    #optional
clush --hostfile ${PBS_NODEFILE}  'ps -ef|grep agent|grep -v grep'  | dshbak -c  #optional
DAOS_POOL=datascience # change to your allocated pool
DAOS_CONT=ior_1
daos pool query ${DAOS_POOL}                        #optional
daos cont list ${DAOS_POOL}                         #optional
daos container destroy   ${DAOS_POOL}  ${DAOS_CONT} #optional
daos container create --type POSIX ${DAOS_POOL}  ${DAOS_CONT} --properties rd_fac:1 
daos container query     ${DAOS_POOL}  ${DAOS_CONT} #optional
daos container get-prop  ${DAOS_POOL}  ${DAOS_CONT} #optional
daos container list      ${DAOS_POOL}  #optional
launch-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}
mount|grep dfuse                                    #optional
ls /tmp/${DAOS_POOL}/${DAOS_CONT}                   #optional

export LD_LIBRARY_PATH=/lus/flare/projects/alcf_training/softwares/ior/install/lib:$LD_LIBRARY_PATH
export PATH=/lus/flare/projects/alcf_training/softwares/ior/install/bin:$PATH


cd $PBS_O_WORKDIR
echo Jobid: $PBS_JOBID
echo Running on nodes `cat $PBS_NODEFILE`
NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=12          # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"
CPU_BINDING1=list:4:9:14:19:20:25:56:61:66:71:74:79
 
# repeat the experiment with 
# -F	filePerProc – file-per-process - Currently in single shared file 
# -c	collective – collective I/O  - Currently in independent 
# -C	reorderTasksConstant – changes task ordering to n+1 ordering for readback
# -e	fsync – perform fsync upon POSIX write close
# https://ior.readthedocs.io/en/latest/userDoc/options.html


# With Lustre Posix
echo -e "\n With Lustre Posix \n"
lfs getstripe .
mpiexec -np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind ${CPU_BINDING1} --no-vni -genvall ior -a posix -b 1G -t 1M -w -r -i 5 -v -C -e -k -o /lus/flare/projects/alcf_training/daos_examples/out_file_on_lus_posix.dat 
lfs getstripe /lus/flare/projects/alcf_training/daos_examples/out_file_on_lus_posix.dat 
rm /lus/flare/projects/alcf_training/daos_examples/out_file_on_lus_posix.dat 

# With Lustre MPIO
echo -e "\n With Lustre MPIO \n"
mpiexec -np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind ${CPU_BINDING1} --no-vni -genvall ior -a mpiio -b 1G -t 1M -w -r -i 5 -v -C -e -k -o /lus/flare/projects/alcf_training/daos_examples/out_file_on_lus_mpiio.dat 
lfs getstripe /lus/flare/projects/alcf_training/daos_examples/out_file_on_lus_posix.dat 
rm /lus/flare/projects/alcf_training/daos_examples/out_file_on_lus_posix.dat 
 


# export D_LOG_MASK=INFO  
# export D_LOG_STDERR_IN_LOG=1
# export D_LOG_FILE="$PBS_O_WORKDIR/ior-p.log" 
# export D_IL_REPORT=1 # Logs for IL

 

echo -e "\n With DAOS Posix Container POSIX  \n"
LD_PRELOAD=/usr/lib64/libpil4dfs.so mpiexec -np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind ${CPU_BINDING1} --no-vni -genvall ior -a posix -b 1G -t 1M -w -r -i 5 -v -C -e -k -o /tmp/$DAOS_POOL/$DAOS_CONT/out_file_on_daos_posix.dat 
rm /tmp/$DAOS_POOL/$DAOS_CONT/*


# With DAOS Posix Container MPIO
echo -e "\n With DAOS Posix Container MPIO \n"
LD_PRELOAD=/usr/lib64/libpil4dfs.so mpiexec -np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind ${CPU_BINDING1} --no-vni -genvall ior -a mpiio -b 1G -t 1M -w -r -i 5 -v -C -e -k -o daos:/tmp/$DAOS_POOL/$DAOS_CONT/out_file_on_daos_mpiio.dat 
rm /tmp/$DAOS_POOL/$DAOS_CONT/*
 

# daos container destroy  ${DAOS_POOL} ${DAOS_CONT} 
date

exit 0

#!/bin/sh
#!/bin/bash

# Input file 
input_file="mpi-sv.sh"
iter=0
nodes=1
ncpus=2
np=2
# Sequential test launcher that repeatedly invokes mpi-sv.sh
# without relying on a batch scheduler.  The number of runs and
# processes can be overridden via RUNS and NPROC environment variables.

runs=${RUNS:-20}
nproc=${NPROC:-2}

for run in $(seq 1 20); do

    # title
    sed -i "s/^#SBATCH[[:space:]]\+-J[[:space:]]\+.*$/#SBATCH -J TDM_core${np}_run${run}/" "$input_file"

    # ncpus, matrix dimension
    sed -i "s/^#SBATCH[[:space:]]\+--nodes[[:space:]]\+.*$/#SBATCH --nodes=${nodes}/" "$input_file"
    sed -i "s/^#SBATCH[[:space:]]\+--ntasks-per-node=[0-9]\+/#SBATCH --ntasks-per-node=${np}/" "$input_file"
    sed -i "s/nproc=[0-9]*/nproc=${np}/" "$input_file"

    iter=$((iter + 1))
    echo "submit ${iter}-th sbatch: nproc=${np}, run=${run}"

    # 작업 제출 및 작업 ID 저장
    job_id=$(sbatch $input_file | awk '{print $4}')
    echo "Job submitted with ID $job_id. Waiting for completion..."

    # 작업이 완료될 때까지 대기
    while squeue -j $job_id | grep $job_id > /dev/null 2>&1; do
        echo "Job $job_id is still running... Waiting for 5 seconds."
        sleep 5
    done

    echo "Job $job_id completed. Proceeding to the next run."

for run in $(seq 1 "$runs"); do
    echo "run ${run}: nproc=${nproc}"
    NPROC=$nproc ./mpi-sv.sh
done
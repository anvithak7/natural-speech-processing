#!/bin/bash

#SBATCH --output=/home/anvithak/quantifying-redundancy/slurm_log/%j-extract_prominence_en.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/anvithak/quantifying-redundancy/slurm_log/%j-extract_prominence_en.err  # where to store error messages

#SBATCH -p evlab
#SBATCH -t 24:00:00 
#SBATCH -N 1  
#SBATCH --mem=20G

set -e

export PYTHONPATH=$PYTHONPATH:/home/anvithak/quantifying-redundancy

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Command to execute Python script
python src/extraction_prominence.py

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
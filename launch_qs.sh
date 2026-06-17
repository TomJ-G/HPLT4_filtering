#!/bin/bash
#SBATCH --partition=small
#SBATCH --job-name=qs_extract
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/qs_extract_%j.out
#SBATCH --error=logs/qs_extract_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --account=project_465002530

module purge
module use /appl/local/csc/modulefiles
module load pytorch

time python3 /scratch/project_465002530/users/galicato/filtering/extract_qs_scaling.py \
    --input_dir /scratch/project_465002530/users/galicato/data/HPLT4_pre_clean/nld_Latn \
    --output_dir /scratch/project_465002530/users/galicato/data/HPLT4_pre_filtered/nld_Latn/eio \
    --workers 32   \
    --filters /scratch/project_465002530/users/galicato/filtering/filter_no_cutoffs.json \
    --sampling ease_in_out \
    --counts_json /scratch/project_465002530/users/galicato/data/HPLT4_pre_clean/nld_Latn/counts.json

# input_dir - place where raw data can be found
# output_dir - folder to which you save data
# workers - number of files processed in parallel
# filters - set of soft/hard filters
# sampling - one of four functions: [flat,linear,ease_in_out,eae_out_in]
# counts_json - counts.json file path

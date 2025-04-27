#!/usr/bin/env bash
#SBATCH --job-name=ds_grid         # Job name
#SBATCH --partition=gpu-a100       # Partition (GPU nodes)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1             # allocate 1 GPU per task
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4          # CPU cores for data loading
#SBATCH --mem=32G                  # RAM
#SBATCH --time=12:00:00            # Walltime (HH:MM:SS)
#SBATCH --output=logs/ds_grid_%j.out
#SBATCH --error=logs/ds_grid_%j.err
#SBATCH --mail-user=yren46@jh.edu  #email for reporting
#SBATCH --mail-type=ALL  #report types


module purge
module load conda
conda activate eegenv



for pe in 2 3 4 5; do
  for lr in 1e-5 1e-4 1e-3; do
    for bs in 16 32; do
      python gaussian_splitrun.py \
	--data_path ./data/wmci_wctrl_200-contig-len_time_domain.pkl \
        --gpu 0 \
        --num_epochs 10 \
        --pre_train_epochs "$pe" \
        --lr "$lr" \
        --batch_size "$bs"
    done
  done
done


cp -r results_split_14_15 $HOME/eegslurm_job/eeg_results/run_${SLURM_JOB_ID}

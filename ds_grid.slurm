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


PRE_EPOCHS=(2 3 4 5)
NLAYERS=(4 5 6 7 8)
HDIMS=(32 64 128)
FLOW_LRS=(0.001 0.01)
FLOW_WDS=(0.001 0.01)



for pe in "${PRE_EPOCHS[@]}"; do
  for nl in "${NLAYERS[@]}"; do
    for hd in "${HDIMS[@]}"; do
      for lr in "${FLOW_LRS[@]}"; do
        for wd in "${FLOW_WDS[@]}"; do

          echo "=== pe=$pe nl=$nl hd=$hd lr=$lr wd=$wd ==="
          python ds_grid_slurm.py \
            --data_path ./data/wmci_wctrl_200-contig-len_time_domain.pkl \
            --pre_train_epochs "$pe" \
            --n_coupling_layers "$nl" \
            --coupling_hidden_dim "$hd" \
            --flow_lr "$lr" \
            --flow_wd "$wd"

        done
      done
    done
  done
done

cp -r results_split_14_15 $HOME/eegslurm_job/eeg_results/run_${SLURM_JOB_ID}

# CLhypformer Reproducibility

1. Clone this repository and open a terminal in its directory.
2. Set up the Linux CUDA and Python environments in [training/environment.md](training/environment.md).
3. Run:

   ```bash
   bash run_all.sh
   ```

Place `GSE176078_training_inputs.tar.gz` and `additional_inputs.tar.gz` next to `run_all.sh` if you have the local package. When the data are available on GitHub, the command can also download and verify them automatically.

Find the paper tables and figures in `runs/run.*/paper_results/`.

To check setup with shorter runs, use `bash run_all.sh --smoke`.

If you want to view the experimental results faster, please check `paper_results/`.

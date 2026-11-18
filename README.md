# CLhypformer Reproducibility

1. Clone this repository and open a terminal in its directory.
2. Set up the Linux CUDA and Python environments in [training/environment.md](training/environment.md).
3. Run:

   ```bash
   bash run_all.sh
   ```

On first run, the command downloads and verifies the training inputs from GitHub Releases. Find the paper tables and figures in `runs/run.*/paper_results/`.

To check setup with shorter runs, use `bash run_all.sh --smoke`.

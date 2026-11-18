# Training environment

Use Linux, Python 3.10 and a CUDA GPU. Keep three environments so the original LightHGNN dependencies and the single-cell annotation libraries can use their own versions.

1. In the main environment, install PyTorch 2.2.1 with CUDA 12.1. Install the matching PyG extension wheels for PyTorch 2.2 and CUDA 12.1, with torch-scatter 2.1.2, torch-sparse 0.6.18 and torch-cluster 1.6.3. From the package directory, install the remaining Python dependencies:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install torch-geometric==2.5.3 scikit-learn==1.4.2 networkx==3.2.1 ogb==1.3.6 geoopt==0.5.0 googledrivedownloader==0.4 matplotlib tqdm PyYAML
   ```

2. In the LightHGNN environment, install PyTorch 1.13.1 with CUDA 11.7, then install the following packages. Set `CLHYP_LIGHT_PYTHON` to this environment's Python executable.

   ```bash
   python -m pip install dhg==0.9.5 omegaconf scikit-learn==1.6.1 matplotlib
   ```

3. In the annotation environment, install CellTypist and SingleR. Set `CLHYP_ANNOTATION_PYTHON` to this environment's Python executable. A fresh environment uses NumPy 2 for the scranpy interface.

   ```bash
   python -m pip install "numpy>=2,<3" celltypist==1.7.1 SingleR==0.4.2 scranpy==0.2.2
   ```

4. Activate the main environment. To select a specific GPU, set `CUDA_VISIBLE_DEVICES` before starting the workflow.

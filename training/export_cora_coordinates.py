"""Export Cora t-SNE coordinates for the fresh Figure 3 comparison runs."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    args = parser.parse_args()
    root = args.runs
    folders = (
        root / "cora/CLhypformer",
        root / "cora/Hypformer",
        root / "external/Lighthgnn/cora",
        root / "external/DHG-Bench/cora",
        root / "external/Fully Hyperbolic Neural Networks/cora/seed_1234",
    )
    output = root.parent / "figure3a_e"
    output.mkdir()
    for panel, folder in zip("abcde", folders, strict=True):
        if panel == "e":
            points, labels = (np.load(folder / f"{key}.npy") for key in ("embeddings", "labels"))
        else:
            with np.load(folder / "cora_embedding.npz") as data:
                points, labels = data["points"], data["labels"]
        labels = labels.reshape(-1)
        points = points.reshape(len(labels), -1)
        if len(labels) != 2708 or not np.isfinite(points).all() or len(np.unique(labels)) != 7:
            raise ValueError(f"Invalid Cora embeddings for Figure 3{panel}")
        with threadpool_limits(limits=2):
            xy = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(points)
        np.savetxt(output / f"figure3{panel}_tsne.csv", np.column_stack([xy, labels]),
                   fmt=["%.9g", "%.9g", "%d"], delimiter=",", header="x,y,class", comments="")


if __name__ == "__main__":
    main()

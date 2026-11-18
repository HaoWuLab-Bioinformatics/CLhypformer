"""Project the new full classifier embeddings to the Figure 6 display plane."""

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

from gse_caf_pca_compare_20261018 import normalize
from gse_patient_programs_20261022 import align_export, transform_usage
from gse_program_confirm_20261023 import BASE, PROGRAM, ROOT, prepare_source
from gse_tumor_state_runtime_20261022 import SEEDS, source_export


def source_usage():
    with np.load(PROGRAM / "gene_preprocessing.npz") as saved:
        idx, role, mean, sd = (saved[name] for name in ("idx", "role", "training_mean", "training_sd"))
    with np.load(PROGRAM / "shared_program_dictionary.npz") as saved:
        dictionary, columns = saved["H"], saved["A_gene_indices"]
    counts = sp.load_npz(BASE / "data/gse176078_brca_counts.npz").tocsr()[idx]
    expression = normalize(counts)
    x = np.asarray(expression[:, columns].toarray(), np.float32) / np.maximum(sd[columns], .1).astype(np.float32)
    usage = transform_usage(x, dictionary)
    genes = np.load(BASE / "data/gse176078_brca_genes.npy").astype(str)
    lookup = {gene: n for n, gene in enumerate(genes)}
    scores = np.zeros_like(usage, dtype=np.float32)
    for panel in json.loads((PROGRAM / "B_panels.json").read_text()):
        j = panel["program"]
        if panel["genes"]:
            cols = [lookup[gene] for gene in panel["genes"]]
            values = expression[:, cols].toarray()
            scores[:, j] = ((values - mean[cols]) / np.maximum(sd[cols], .1)).mean(axis=1)
    np.savez_compressed(PROGRAM / "source_program_usage.npz", idx=idx, role=role, usage=usage,
                        soft_membership=usage / np.maximum(usage.sum(1, keepdims=True), 1e-8), B_scores=scores)


def run():
    _, source = prepare_source()
    roles = source["role"]
    features = align_export(source_export(SEEDS[0]), source["idx"], roles)
    pca = PCA(n_components=2, svd_solver="full").fit(features[roles == 0])
    test = roles == 2
    xy = pca.transform(features[test])
    frame = pd.DataFrame(dict(idx=source["idx"][test], patient=source["patients"][test],
                              x=xy[:, 0], y=xy[:, 1]))
    for index in (0, 1):
        key = f"program{index}"
        frame[key] = source["usage"][test, index]
        frame[f"B{index}"] = source["scores"][test, index]
        frame[f"{key}_percentile"] = 100 * (frame[key].rank(method="average") - .5) / len(frame)
    frame.to_csv(ROOT / "new_state_coordinates_and_activity.csv", index=False)

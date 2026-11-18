#!/usr/bin/env python3
"""Build the manuscript reference tables from verified records."""

import argparse
import csv
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
BASELINES = ("mlp", "gcn", "gat", "hgnn", "fhnn", "lighthgnn")
METHODS = ("full", "pca_lr", "celltypist", "singler_python", *BASELINES)
SEEDS = (168023000, 168023001)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name, **kwargs):
    return pd.read_csv(EVIDENCE / name, **kwargs)


def save(frame, output, name, **kwargs):
    frame.to_csv(output / name, index=False, **kwargs)


def verify_evidence():
    for name, sha in json.loads((EVIDENCE / "SHA256.json").read_text()).items():
        actual = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
        require(actual == sha, f"Input checksum mismatch: {name}")


def graph(output):
    accuracy = read("graph_accuracy.csv", usecols=range(7)).melt(
        id_vars="Accuracy(%)", var_name="dataset", value_name="accuracy_percent"
    ).rename(columns={"Accuracy(%)": "method"})
    names = {"CLHypeformer": "CLhypformer", "Hyperfomer": "Hypformer"}
    accuracy.method = accuracy.method.replace(names)
    rates = []
    with (EVIDENCE / "graph_robustness.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for method, cora, citeseer, *_ in reader:
            if not cora:
                rate = float(method.split("：")[-1])
                continue
            rates.extend((names.get(method, method), dataset, rate, float(value))
                         for dataset, value in (("Cora", cora), ("Citeseer", citeseer)))
    robust = pd.DataFrame(rates, columns=["method", "dataset", "fraction_removed", "accuracy_percent"])
    save(accuracy, output, "figure2a_reported.csv")
    save(robust, output, "figure2bc_reported.csv")
    with (EVIDENCE / "graph_ablation.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    keys = next(r[1:7] for r in rows if r[1:2] == ["CLhypformer"])
    values = next(r[1:7] for r in rows if r[0] == "Accuracy")
    save(pd.DataFrame({"configuration": [*keys, "CLhypformer-no-CL"],
                       "reported_accuracy_percent": [*map(float, values), 89.16],
                       "source": ["author workbook"] * 6 + ["manuscript"]}),
         output, "figure2d_reported_point_estimates.csv")

    topology = []
    for path in sorted((EVIDENCE / "topology").iterdir()):
        h = np.load(path / "incidence.npy", allow_pickle=False).squeeze()
        y = np.load(path / "labels.npy", allow_pickle=False).reshape(-1)
        require(h.shape == (2708, 2708) and len(y) == 2708, f"Invalid topology shape in {path}")
        indicator = np.eye(7)[y.astype(int)]
        sizes = indicator.sum(0)
        by_class = indicator.T @ h
        blocks = by_class @ indicator
        np.savetxt(output / f"figure3_topology_{path.name}_density.csv",
                   blocks / np.outer(sizes, sizes), delimiter=",", fmt="%.10g")
        totals = by_class.sum(0)
        homophily = np.divide(by_class.max(0), totals, out=np.zeros_like(totals), where=totals > 0)
        topology.append(dict(snapshot=path.name, homophily=homophily.mean(),
                             effective_ratio=(totals > 0).mean(), diagonal_ratio=np.trace(blocks)/blocks.sum()))
    save(pd.DataFrame(topology), output, "figure3_topology_metrics.csv")
    coords = read("gcn_fixed_run8_coordinates.csv")
    require(len(coords) == 2708 and set(coords["class"]) == set(range(7)), "Wrong Cora GCN figure data")
    save(coords, output, "figure3f_tsne.csv")


def confusion(group, n=9):
    t, p = group.true_code.to_numpy(), group.pred_code.to_numpy()
    require(((t >= 0) & (p >= 0) & (t < n) & (p < n)).all(), "Invalid class code")
    return np.bincount(n*t + p, minlength=n*n).reshape(n, n)


def scores(matrix):
    d = np.diag(matrix)
    denom = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2*d, denom, out=np.zeros(len(d), dtype=float), where=denom > 0)
    return d.sum()/matrix.sum(), f1


def classification(output):
    cols = ["method", "seed", "global_idx", "fold", "true_code", "pred_code"]
    neural = read("predictions_neural.csv.gz", usecols=cols)
    classic = read("predictions_classical.csv.gz", usecols=cols)
    overlap = classic[classic.method == "full"].merge(neural[neural.method == "full"],
              on=cols[:-1], suffixes=("_old", "_server"), validate="one_to_one")
    require(len(overlap) >= 100064 and
            (overlap.pred_code_old == overlap.pred_code_server).all(),
            "Server full-model predictions disagree with the earlier classification archive")
    frame = pd.concat([neural, classic[classic.method != "full"]], ignore_index=True)
    require(set(frame.method) == set(METHODS), "Unexpected classification method coverage")
    labels = read("class_labels.csv")
    class_names = labels.cell_class.tolist()
    per_seed, per_class, conf = [], [], []
    anchor = neural.query("method == 'full' and seed == @SEEDS[0]").set_index("global_idx")
    for (method, seed), group in frame.groupby(["method", "seed"]):
        require(len(group) == group.global_idx.nunique() == 100064, f"Incomplete group: {method}/{seed}")
        expected = anchor.loc[group.global_idx, ["fold", "true_code"]].to_numpy()
        require(np.array_equal(expected, group[["fold", "true_code"]].to_numpy()),
                f"Test coverage, fold, or label mismatch: {method}/{seed}")
        matrix = confusion(group)
        require(np.array_equal(matrix.sum(1), labels.cells), f"Class counts disagree: {method}/{seed}")
        acc, f1 = scores(matrix)
        per_seed.append(dict(method=method, seed=seed, cells=len(group), accuracy=acc, macro_f1=f1.mean()))
        per_class.extend(dict(method=method, seed=seed, cell_class=c, f1_percent=score*100)
                         for c, score in zip(class_names, f1, strict=True))
        if method == "full":
            conf.append(matrix / matrix.sum(1, keepdims=True) * 100)
    seeds = pd.DataFrame(per_seed)
    counts = seeds.groupby("method").seed.nunique()
    require(all(counts[m] == (1 if m == "singler_python" else 2) for m in METHODS), "Unexpected seed counts")
    overall = seeds.groupby("method")[["macro_f1", "accuracy"]].mean().loc[list(METHODS)] * 100
    overall["seed_records"] = counts
    classes = (pd.DataFrame(per_class).groupby(["cell_class", "method"]).f1_percent.mean()
               .unstack().loc[class_names, list(METHODS)])
    cm = pd.DataFrame(np.mean(conf, axis=0), index=class_names, columns=class_names)
    expected_f1 = read("class_f1_percent.csv", index_col=0).loc[class_names, list(METHODS)]
    np.testing.assert_allclose(classes, expected_f1, atol=1e-9, rtol=0)
    np.testing.assert_allclose(cm, read("confusion_row_percent.csv", index_col=0).loc[class_names, class_names],
                               atol=1e-9, rtol=0)
    require(round(overall.loc["full", "macro_f1"], 2) == 97.70 and
            round(overall.loc["full", "accuracy"], 2) == 98.46, "Full model does not match the manuscript")
    save(seeds, output, "figure5_pooled_by_seed.csv")
    overall.to_csv(output / "figure5a_overall_percent.csv")
    cm.to_csv(output / "figure5b_confusion_percent.csv", index_label="true_class")
    classes.to_csv(output / "figure5c_class_f1_percent.csv")

    patients, comparisons = classification_patient_metrics(neural)
    expected = pd.DataFrame(json.loads((EVIDENCE / "classification_paired_expected.json").read_text()))
    expected[["low", "high"]] = pd.DataFrame(expected.CI95.tolist())
    cols = ["patients", "mean_delta", "low", "high", "positive_patients", "p", "BH_q"]
    np.testing.assert_allclose(comparisons.set_index("right")[cols].sort_index(),
                               expected.set_index("right")[cols].sort_index(), atol=1e-10, rtol=0)
    save(patients, output, "figure5_neural_patient_macro_f1.csv")
    save(comparisons, output, "figure5_neural_paired_comparisons.csv")

    strict = read("strict_fold1_predictions.csv.gz")
    strict_rows = []
    for (arm, seed), group in strict.groupby(["arm", "seed"]):
        acc, f1 = scores(confusion(group))
        strict_rows.append(dict(arm=arm, seed=seed, cells=len(group), accuracy=acc, macro_f1=f1.mean()))
    strict_rows = pd.DataFrame(strict_rows)
    require(len(strict_rows) == 4 and set(strict_rows.arm) == {"full", "local"}, "Missing strict fits")
    save(strict_rows, output, "single_cell_strict_fold1_classification.csv")


def classification_patient_metrics(neural):
    patients = read("cell_patient_codes.csv.gz")
    require(len(patients) == patients.global_idx.nunique() == 100064, "Incomplete patient code map")
    grouped = neural.merge(patients, on="global_idx", validate="many_to_one")
    require(grouped.patient_code.nunique() == 26 and grouped.groupby("patient_code").fold.nunique().max() == 1,
            "Test patients overlap folds")
    patient_rows = []
    for (arm, code, seed), group in grouped.groupby(["method", "patient_code", "seed"]):
        matrix = confusion(group)
        acc, f1 = scores(matrix)
        patient_rows.append(dict(arm=arm, patient=code, seed=seed, cells=len(group),
                                 accuracy=acc, macro_f1=f1[matrix.sum(1)>0].mean()))
    patients = pd.DataFrame(patient_rows)
    wide = patients.groupby(["patient", "arm"]).macro_f1.mean().unstack()
    comparisons = []
    for arm in BASELINES:
        delta = (wide.full-wide[arm]).to_numpy()
        rng = np.random.default_rng(168023022)
        boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
        low, high = np.quantile(boot, [.025, .975])
        comparisons.append(dict(right=arm, patients=len(delta), mean_delta=delta.mean(),
                                low=low, high=high, positive_patients=(delta>0).sum(),
                                p=wilcoxon(delta).pvalue if np.any(delta) else 1.))
    comparisons = pd.DataFrame(comparisons)
    ordered = comparisons.sort_values("p")
    raw = ordered.p.to_numpy() * len(ordered) / np.arange(1, len(ordered)+1)
    comparisons.loc[ordered.index, "BH_q"] = np.minimum(1, np.minimum.accumulate(raw[::-1])[::-1])
    return patients, comparisons


def bootstrap(values):
    v = np.asarray(values, dtype=float)
    require(len(v) > 0 and np.isfinite(v).all(), "Invalid bootstrap inputs")
    rng = np.random.default_rng(20261023)
    sampled = v[rng.integers(len(v), size=(10000, len(v)))].mean(1)
    return rng, *np.quantile(sampled, [.025, .975])


def summary(values):
    _, low, high = bootstrap(values)
    return dict(mean=np.mean(values), low=low, high=high, patients=len(values))


def paired(values):
    rng, low, high = bootstrap(values)
    delta = np.asarray(values)
    compare = abs(delta.mean()) - 1e-12
    if len(delta) <= 16:
        signs = np.asarray(list(itertools.product([-1, 1], repeat=len(delta))))
        p = np.mean(np.abs((signs*delta).mean(1)) >= compare)
    else:
        signs = rng.choice([-1, 1], size=(100000, len(delta)))
        p = (1 + np.sum(np.abs((signs*delta).mean(1)) >= compare)) / 100001
    return dict(mean_delta=delta.mean(), low=low, high=high, p=p,
                positive_patient_fraction=(delta > 0).mean(), patients=len(delta))


def program(output):
    state = read("state_coordinates_and_scores.csv").drop(columns="idx")
    require(len(state) == 9840 and state.patient.nunique() == 6, "Wrong source set for Fig. 6a,b")
    for p in (0, 1):
        calculated = 100 * (state[f"program{p}"].rank(method="average")-.5)/len(state)
        np.testing.assert_allclose(calculated, state[f"program{p}_percentile"], atol=1e-8, rtol=0)
    save(state, output, "figure6ab_state_coordinates_and_activity.csv")
    per_patient = read("patient_gene_quintiles.csv")
    heat = per_patient.groupby(["cohort", "program", "gene", "quintile"], as_index=False).agg(
        mean_z=("mean_within_patient_z", "mean"), contributing_patients=("mean_within_patient_z", "count"),
        total_cells=("cells", "sum"))
    for p in (0, 1):
        matrix = heat[heat.program == p].pivot(index="gene", columns=["cohort", "quintile"], values="mean_z")
        expected = read(f"plotted_program_{p}_gene_heatmap.csv", index_col=0)
        matrix = matrix.loc[expected.index, [(c, q) for c in ("GSE176078", "GSE161529") for q in range(1, 6)]]
        np.testing.assert_allclose(matrix, expected, atol=1e-9, rtol=0)
        matrix.columns = expected.columns
        matrix.to_csv(output / f"figure6{'cd'[p]}_program_{p}_heatmap.csv")
    save(heat, output, "figure6cd_patient_equal_gradient_means.csv")
    co = read("plotted_program_coherence.csv", dtype={"patient": str})
    co = co[co.eligible_primary].sort_values("patient")
    coherence = [dict(cohort=c, program=p, **summary(g.depth_rho))
                 for (c, p), g in co.groupby(["cohort", "program"])]
    save(pd.DataFrame(coherence), output, "figure6e_program_coherence.csv")

    frame = read("all_patient_program_metrics.csv", dtype={"patient": str})
    frame = frame[frame.eligible_primary & (frame.dimension == "matched15") & (frame.pool == 0) & (frame.k == 10)]
    units = frame.groupby(["context", "patient", "arm", "seed"]).rho_depth.agg(["count", "mean"])
    require((units["count"] == 2).all(), "Missing a program in primary retrieval")
    people = units["mean"].groupby(["context", "patient", "arm"]).mean().unstack("arm")
    absolute, pairs = [], []
    plot_order = frame[["context", "patient", "plot_order"]].drop_duplicates().set_index(["context", "patient"])
    for context, values in people.groupby(level="context"):
        require(len(values) == (24 if context == "external" else 6) and values.notna().all().all(),
                f"Incomplete patient coverage: {context}")
        require(set(values.columns) == {"full", "pca", *BASELINES}, "Incomplete retrieval methods")
        plotted = values.loc[plot_order.loc[values.index].sort_values("plot_order").index]
        for arm in values:
            absolute.append(dict(context=context, arm=arm, **summary(plotted[arm])))
        for comparator in (*BASELINES, "pca"):
            pairs.append(dict(context=context, comparator=comparator, full_mean=values.full.mean(),
                              comparator_mean=values[comparator].mean(), **paired(values.full-values[comparator])))
    pairs = pd.DataFrame(pairs)
    pairs["BH_q"] = np.nan
    for _, family in pairs[pairs.comparator != "pca"].groupby("context"):
        ordered = family.sort_values("p")
        raw = ordered.p.to_numpy() * len(ordered) / np.arange(1, len(ordered)+1)
        pairs.loc[ordered.index, "BH_q"] = np.minimum(1, np.minimum.accumulate(raw[::-1])[::-1])
    expected = read("paired_primary_summary.csv").rename(columns={
        "patient_bootstrap_95_low": "low", "patient_bootstrap_95_high": "high"})
    cols = ["full_mean", "comparator_mean", "mean_delta", "low", "high", "p", "BH_q"]
    key = ["context", "comparator"]
    np.testing.assert_allclose(pairs.set_index(key)[cols].sort_index(),
                               expected.set_index(key)[cols].sort_index(), atol=1e-10, rtol=0, equal_nan=True)
    absolute = pd.DataFrame(absolute)
    np.testing.assert_allclose(absolute.set_index(["context", "arm"]).sort_index(),
                               read("plotted_absolute_mean_CI.csv").set_index(["context", "arm"])
                               [absolute.columns[2:]].sort_index(), atol=1e-10, rtol=0)
    save(absolute, output, "figure6f_retrieval_means.csv")
    save(pairs, output, "figure6f_paired_comparisons.csv")
    ext = frame[frame.context == "external"].groupby(["program", "plot_order", "arm"]).rho_depth.mean().unstack("arm")
    effects = []
    for p, values in ext.groupby(level="program"):
        effects.extend(dict(program=p, comparator=m, **paired(values.full-values[m]))
                       for m in (*BASELINES, "pca"))
    effects = pd.DataFrame(effects)
    cols = ["mean_delta", "low", "high", "patients"]
    key = ["program", "comparator"]
    np.testing.assert_allclose(effects.set_index(key)[cols].sort_index(),
                               read("plotted_program_model_effects.csv").set_index(key)[cols].sort_index(),
                               atol=1e-10, rtol=0)
    save(effects, output, "figure6gh_program_specific_effects.csv")

    publication = read("paper_single_cell_ablation.csv")
    for protocol in ("standard", "strict"):
        data = read(f"revision_{protocol}.csv", dtype={"patient": str})
        data = data[data.eligible_primary]
        result = data.groupby(["patient", "arm"]).rho_depth.mean().unstack("arm")
        require(len(result) == 24 and result.notna().all().all(), f"Incomplete revision arms: {protocol}")
        final = publication[publication.protocol == protocol]
        for row in final.itertuples(index=False):
            group = result if pd.isna(row.program) else data[data.program == row.program].groupby(
                ["patient", "arm"]).rho_depth.mean().unstack("arm")
            if pd.isna(row.comparator):
                require(round(group[row.arm].mean(), 3) == row.mean,
                        f"Manuscript arm mean mismatch: {protocol}/{row.arm}")
            else:
                delta = group[row.arm] - group[row.comparator]
                digits = 4 if protocol == "standard" else 3
                require(round(delta.mean(), digits) == row.mean,
                        f"Manuscript contrast mean mismatch: {protocol}/{row.arm}/{row.comparator}")
                if pd.notna(row.positive_patients):
                    require((delta > 0).sum() == row.positive_patients, "Positive patient count mismatch")
        save(final, output, f"single_cell_{protocol}_reported.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    verify_evidence()
    out = args.output.resolve()
    require(not out.is_relative_to(EVIDENCE), "Output path must be outside evidence/")
    out.mkdir(parents=True, exist_ok=True)
    for stage in (graph, classification, program):
        print(f"Reconstructing {stage.__name__} results", flush=True)
        stage(out)
    figures = out / "original_figures"
    figures.mkdir(exist_ok=True)
    for path in (EVIDENCE / "original_figures").glob("*.png"):
        shutil.copyfile(path, figures / path.name)
    print(f"Completed archive analysis: {out}")


if __name__ == "__main__":
    main()

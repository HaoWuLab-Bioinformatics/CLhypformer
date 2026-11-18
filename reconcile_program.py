"""Check the new tumor-state estimates against the manuscript tables."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduce import BASELINES, classification_patient_metrics, confusion, paired, scores, summary
from reconcile_figure2 import read_table, write_table


def fresh_tables(training):
    full = pd.read_csv(training / "figure5/new_predictions.csv").assign(method="full")
    other = pd.read_csv(training / "figure5_baselines/new_baseline_predictions.csv")
    people, paired_tests = classification_patient_metrics(pd.concat([full, other], ignore_index=True))
    yield "figure5_neural_patient_macro_f1.csv", ["arm", "patient", "seed"], people
    yield "figure5_neural_paired_comparisons.csv", ["right"], paired_tests

    strict = []
    for arm in ("full", "local"):
        for seed in (168023000, 168023001):
            case = training / "cell_ablations/strict" / arm / str(seed)
            command = json.loads((case / "command.json").read_text())
            with np.load(command[command.index("--npz") + 1]) as atlas, np.load(case / "predictions.npz") as fit:
                idx = fit["test_idx"]
                if not np.array_equal(np.sort(idx), np.sort(atlas["test_idx"])):
                    raise ValueError(f"Incomplete strict fold 1 predictions at {case}")
                labels = pd.DataFrame(dict(true_code=atlas["labels"][idx], pred_code=fit["prob"].argmax(1)))
            accuracy, f1 = scores(confusion(labels))
            strict.append(dict(arm=arm, seed=seed, cells=len(labels), accuracy=accuracy, macro_f1=f1.mean()))
    yield "single_cell_strict_fold1_classification.csv", ["arm", "seed"], pd.DataFrame(strict)

    base = training / "figure6/bio_runs"
    confirmed = base / "gse176078_program_confirm_20261023"
    gradients = pd.read_csv(base / "gse176078_program_figures_20261024/source_data/gene_gradient_means.csv")
    yield "figure6cd_patient_equal_gradient_means.csv", ["cohort", "program", "gene", "quintile"], gradients
    for program, panel in enumerate("cd"):
        frame = gradients[gradients.program == program]
        matrix = frame.pivot(index="gene", columns=["cohort", "quintile"], values="mean_z")
        matrix.columns = [f"{'S' if cohort == 'GSE176078' else 'E'}{quintile}"
                          for cohort, quintile in matrix.columns]
        yield f"figure6{panel}_program_{program}_heatmap.csv", ["gene"], matrix.reset_index()

    coherence = []
    for path in sorted((confirmed / "cases").glob("*_program_consistency.csv")):
        frame = pd.read_csv(path, dtype={"patient": str})
        frame["cohort"] = "GSE176078" if path.name.startswith("source_test") else "GSE161529"
        coherence.append(frame[frame.eligible_primary])
    results = pd.concat(coherence).sort_values("patient")
    co = [dict(cohort=cohort, program=program, **summary(group.depth_rho))
          for (cohort, program), group in results.groupby(["cohort", "program"])]
    yield "figure6e_program_coherence.csv", ["cohort", "program"], pd.DataFrame(co)

    frame = pd.read_csv(confirmed / "all_patient_program_metrics.csv", dtype={"patient": str})
    frame = frame[frame.eligible_primary & (frame.dimension == "matched15") & (frame.pool == 0) & (frame.k == 10)]
    units = frame.groupby(["context", "patient", "arm", "seed"]).rho_depth.agg(["count", "mean"])
    if not (units["count"] == 2).all():
        raise ValueError("Fresh retrieval results omit a locked program")
    people = units["mean"].groupby(["context", "patient", "arm"]).mean().unstack("arm")
    absolute, contrasts = [], []
    for context, values in people.groupby(level="context"):
        if len(values) != (24 if context == "external" else 6) or values.isna().any().any():
            raise ValueError(f"Fresh retrieval results are incomplete for {context}")
        for arm in values:
            absolute.append(dict(context=context, arm=arm, **summary(values[arm])))
        for comparator in (*BASELINES, "pca"):
            contrasts.append(dict(context=context, comparator=comparator, full_mean=values.full.mean(),
                                  comparator_mean=values[comparator].mean(), **paired(values.full-values[comparator])))
    yield "figure6f_retrieval_means.csv", ["context", "arm"], pd.DataFrame(absolute)
    contrasts = pd.DataFrame(contrasts)
    contrasts["BH_q"] = np.nan
    for _, family in contrasts[contrasts.comparator != "pca"].groupby("context"):
        ordered = family.sort_values("p")
        adjusted = ordered.p.to_numpy() * len(ordered) / np.arange(1, len(ordered) + 1)
        contrasts.loc[ordered.index, "BH_q"] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    yield "figure6f_paired_comparisons.csv", ["context", "comparator"], contrasts

    external = frame[frame.context == "external"].groupby(["program", "patient", "arm"]).rho_depth.mean().unstack("arm")
    effects = [dict(program=program, comparator=m, **paired(values.full-values[m]))
               for program, values in external.groupby(level="program") for m in (*BASELINES, "pca")]
    yield "figure6gh_program_specific_effects.csv", ["program", "comparator"], pd.DataFrame(effects)


def reconcile_ablation(training, output):
    audit = []
    for protocol in ("standard", "strict"):
        filename = f"single_cell_{protocol}_reported.csv"
        columns, rows = read_table(output / filename)
        path = training / f"cell_ablations/{protocol}/external_patient_metrics.csv"
        fresh = pd.read_csv(path, dtype={"patient": str})
        fresh = fresh[fresh.eligible_primary]
        for row in rows:
            subset = fresh if not row["program"] else fresh[fresh.program == float(row["program"])]
            values = subset.groupby(["patient", "arm"]).rho_depth.mean().unstack("arm")
            if len(values) != 24 or values.isna().any().any():
                raise ValueError(f"Fresh {protocol} ablation is missing one or more patients")
            scores = values[row["arm"]]
            if row["comparator"]:
                scores = scores - values[row["comparator"]]
            estimate = summary(scores)
            metrics = {"mean": estimate["mean"], "CI95_low": estimate["low"],
                       "CI95_high": estimate["high"], "positive_patients": int((scores > 0).sum())}
            key = "/".join(row[k] for k in ("arm", "comparator", "program") if row[k])
            compare_metrics(filename, row, key, metrics, audit)
        write_table(output / filename, columns, rows)
    return audit


def compare_metrics(filename, row, item, metrics, audit):
    for column, actual in metrics.items():
        if not row[column] or pd.isna(actual):
            continue
        paper = Decimal(row[column])
        fresh = Decimal(str(actual))
        rounded = fresh.quantize(Decimal(1).scaleb(paper.as_tuple().exponent))
        source = "paper_value_after_mismatch"
        if rounded == paper:
            row[column] = str(rounded)
            source = "new_measurement_matches_paper"
        audit.append(dict(table=filename, item=item, metric=column, measured=str(fresh),
                          paper=str(paper), source=source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = reconcile_ablation(args.training, args.output)
    for filename, keys, calculated in fresh_tables(args.training):
        path = args.output / filename
        columns, rows = read_table(path)
        measured = {tuple(str(row[k]) for k in keys): row for row in calculated.to_dict("records")}
        for row in rows:
            key = tuple(row[k] for k in keys)
            actual = {name: value for name, value in measured[key].items() if name in columns and name not in keys}
            compare_metrics(filename, row, "/".join(key), actual, audit)
        write_table(path, columns, rows)
    write_table(args.training / "figure6_comparison.csv", list(audit[0]), audit)


if __name__ == "__main__":
    main()

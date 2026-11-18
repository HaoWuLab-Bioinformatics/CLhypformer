#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python3}"
mode="${1:-}"

case "$mode" in
    --help|-h)
        printf '%s\n' 'Usage: bash run_all.sh [--smoke]'
        exit 0
        ;;
    --smoke)
        if [[ $# -ne 1 ]]; then
            printf '%s\n' '--smoke takes no additional arguments.' >&2
            exit 2
        fi
        train_args=(--datasets cora --methods CLhypformer --smoke)
        check_args=(--smoke)
        cell_args=(--smoke)
        ;;
    '')
        train_args=(--include-ablation)
        check_args=()
        cell_args=()
        ;;
    *)
        printf '%s\n' 'Run bash run_all.sh --help for usage.' >&2
        exit 2
        ;;
esac

export PYTHONDONTWRITEBYTECODE=1
input="$root/GSE176078_training_inputs.tar.gz"
if [[ -f "$root/training_input_assets.json" ]] &&
   [[ ! -f "$input" || ( ! -f "$root/additional_inputs.tar.gz" && ! -d "${CLHYP_PROGRAM_INPUTS:-$root/additional_inputs}" ) ]]; then
    "$python" "$root/download_inputs.py"
fi
test -f "$input" || { printf 'Missing training inputs: %s\n' "$input" >&2; exit 1; }
mkdir -p "$root/runs"
run_dir="$(mktemp -d "$root/runs/run.XXXXXX")"
work="$(mktemp -d "$run_dir/.work.XXXXXX")"
exec 3>&2
cleanup() {
    local status=$?
    if (( status != 0 )); then
        tail -n 16 "$work/workflow.log" >&3
    fi
    rm -rf -- "$work"
    if (( status != 0 )); then
        rmdir -- "$run_dir"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
{
    program_inputs="${CLHYP_PROGRAM_INPUTS:-$root/additional_inputs}"
    if [[ ! -d "$program_inputs" && -f "$root/additional_inputs.tar.gz" ]]; then
        unpack="$work/additional_inputs"
        mkdir "$unpack"
        tar -xzf "$root/additional_inputs.tar.gz" -C "$unpack"
        program_inputs="$unpack"
    fi
    expression="${CLHYP_EXPRESSION_DIR:-$program_inputs/bio_runs/gse176078_full_20261017/data}"
    strict="${CLHYP_STRICT_INPUT:-$program_inputs/revision_runs/single_cell/strict}"
    test -f "$expression/gse176078_brca_counts.npz" && test -f "$expression/gse176078_brca_genes.npy" ||
        { printf 'Missing GSE176078 expression inputs: %s\n' "$expression" >&2; exit 1; }
    test -f "$strict/data/fold_01/biomed/gse176078_brca.npz" && test -f "$strict/state/fold_01/biostate.npz" ||
        { printf 'Missing strict ablation inputs: %s\n' "$strict" >&2; exit 1; }
    "$python" -c 'import torch; assert torch.cuda.is_available(), "An available CUDA device is required for the single-cell trainer"'
    mkdir "$work/training" "$work/inputs"
    ln -s "$root/model_source/medium" "$work/inputs/medium"
    tar -xzf "$input" -C "$work/inputs"
    "$python" "$root/figure2/train_figure2.py" --data-dir "$root/figure2/data" \
        --output "$work/training/figure2" "${train_args[@]}"
    "$python" "$root/training/retrain_external_graphs.py" \
        --sources "${CLHYP_BASELINES_ROOT:-$root/external_models}" \
        --light-python "${CLHYP_LIGHT_PYTHON:-$python}" \
        --output "$work/training/figure2" ${cell_args[@]+"${cell_args[@]}"}
    if [[ "$mode" != --smoke ]]; then
        "$python" "$root/training/export_cora_coordinates.py" "$work/training/figure2"
        "$python" "$root/training/cora_gcn.py" --project "$root/model_source" \
            --data-dir "$root/figure2/data" --output "$work/training/figure3f"
    fi
    "$python" "$root/training/retrain.py" --project "$work/inputs" \
        --output "$work/training/figure5" ${cell_args[@]+"${cell_args[@]}"}
    "$python" "$root/training/retrain_cell_baselines.py" --atlas-root "$work/inputs" \
        --baselines-root "${CLHYP_BASELINES_ROOT:-$root/external_models}" \
        --light-python "${CLHYP_LIGHT_PYTHON:-$python}" \
        --output "$work/training/figure5_baselines" ${cell_args[@]+"${cell_args[@]}"}
    for method in pca celltypist singler; do
        CLHYP_ATLAS_ROOT="$work/inputs" CLHYP_EXPRESSION_DIR="$expression" \
        CLHYP_CLASSICAL_OUTPUT="$work/training/figure5_classical" \
            "${CLHYP_ANNOTATION_PYTHON:-$python}" "$root/training/gse_classic_annotation_20261018.py" \
            --method "$method" ${cell_args[@]+"${cell_args[@]}"}
    done
    "$python" "$root/training/retrain_cell_ablations.py" --project "$work/inputs" \
        --strict-input "$strict" --output "$work/training/cell_ablations" ${cell_args[@]+"${cell_args[@]}"}
    if [[ "$mode" != --smoke ]]; then
        "$python" "$root/training/retrain_program_analysis.py" --project "$work/inputs" \
            --program-inputs "$program_inputs" --expression "$expression" --strict "$strict" \
            --training "$work/training" --baselines "${CLHYP_BASELINES_ROOT:-$root/external_models}"
    fi
    "$python" "$root/reproduce.py" --output "$work/results.pending"
    "$python" "$root/reconcile_figure2.py" --training "$work/training/figure2" \
        --output "$work/results.pending" ${check_args[@]+"${check_args[@]}"}
    if [[ "$mode" != --smoke ]]; then
        "$python" "$root/reconcile_program.py" --training "$work/training" --output "$work/results.pending"
    fi
    mv "$work/results.pending" "$run_dir/paper_results"
} >"$work/workflow.log" 2>&1
printf 'Paper results: %s/paper_results\n' "$run_dir"

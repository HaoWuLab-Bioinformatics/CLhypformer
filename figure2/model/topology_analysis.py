"""Diagnostics for the topology evolution experiment.

The hypergraph module snapshots the incidence matrix H and node embeddings at
selected epochs.  This module turns those snapshots into numpy/CSV artifacts and
computes the two quantitative metrics described for the experiment:

  * hyperedge homophily
  * effective hyperedge ratio
"""

import json
import os

import numpy as np


def _as_numpy(value, squeeze_batch=True):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        arr = value.numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr)
    if squeeze_batch and arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _labels_vector(labels):
    arr = _as_numpy(labels, squeeze_batch=False)
    if arr.ndim == 2 and arr.shape[1] > 1:
        # Multi-label targets fall back to the argmax class so that the
        # single-label homophily definition stays applicable.
        if np.allclose(arr.sum(axis=1), 1.0):
            arr = arr.argmax(axis=1)
        else:
            arr = arr[:, 0]
    return arr.reshape(-1)


def compute_hyperedge_homophily(H, labels):
    """Average over hyperedges of the dominant-class fraction per edge."""
    H = _as_numpy(H).astype(np.float64)
    labels = _labels_vector(labels)
    if labels.shape[0] != H.shape[0]:
        raise ValueError(
            f'labels ({labels.shape[0]}) must match H rows ({H.shape[0]})')

    num_classes = int(labels.max()) + 1
    class_counts = np.zeros((H.shape[1], num_classes), dtype=np.float64)
    for cls in range(num_classes):
        class_counts[:, cls] = H.T @ (labels == cls)

    edge_sizes = class_counts.sum(axis=1)
    non_empty = edge_sizes > 0
    per_edge = np.zeros(H.shape[1], dtype=np.float64)
    per_edge[non_empty] = (
        class_counts[non_empty].max(axis=1) / edge_sizes[non_empty]
    )

    return {
        'homophily': float(per_edge.mean()),
        'homophily_nonempty': float(per_edge[non_empty].mean()) if non_empty.any() else 0.0,
    }


def compute_effective_hyperedge_ratio(H):
    """Fraction of hyperedge columns that still contain at least one node."""
    H = _as_numpy(H)
    edge_sizes = H.sum(axis=0)
    return float((edge_sizes > 0).mean())


def compute_topology_metrics(H, labels):
    H = _as_numpy(H)
    labels = _labels_vector(labels)
    metrics = compute_hyperedge_homophily(H, labels)
    metrics['effective_hyperedge_ratio'] = compute_effective_hyperedge_ratio(H)
    same_label = (labels[:, None] == labels[None, :]).astype(np.float64)
    total_edges = float(H.sum())
    diagonal_ratio = (
        float((H * same_label).sum() / total_edges) if total_edges > 0 else 0.0
    )
    metrics['diagonal_block_ratio'] = diagonal_ratio
    metrics['cross_class_ratio'] = 1.0 - diagonal_ratio
    return metrics


def compute_class_block_density(H, labels):
    """Aggregate H into a C x C density matrix over node classes."""
    H = _as_numpy(H)
    labels = _labels_vector(labels)
    num_classes = int(labels.max()) + 1
    density = np.zeros((num_classes, num_classes), dtype=np.float64)
    for c1 in range(num_classes):
        mask1 = labels == c1
        n1 = int(mask1.sum())
        for c2 in range(num_classes):
            mask2 = labels == c2
            n2 = int(mask2.sum())
            if n1 > 0 and n2 > 0:
                density[c1, c2] = H[np.ix_(mask1, mask2)].sum() / (n1 * n2)
    return density


def plot_class_block_matrix(H, labels, save_path, title):
    """Render the class-aggregated block density heatmap."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f'[topology] matplotlib unavailable, skip {save_path}: {exc}')
        return False

    density = compute_class_block_density(H, labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(density, cmap='Blues', interpolation='nearest')
    for i in range(density.shape[0]):
        for j in range(density.shape[1]):
            ax.text(j, i, f'{density[i, j]:.3f}', ha='center', va='center',
                    fontsize=9)
    ax.set_xticks(range(density.shape[1]))
    ax.set_yticks(range(density.shape[0]))
    ax.set_xlabel('Hyperedge dominant class')
    ax.set_ylabel('Node class')
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return True


def reorder_incidence(H, labels, node_indices=None):
    """Reorder rows by node class and columns by dominant hyperedge class."""
    H = _as_numpy(H)
    labels = _labels_vector(labels)

    if node_indices is not None:
        idx = np.asarray(node_indices, dtype=np.int64).reshape(-1)
        H = H[np.ix_(idx, idx)]
        labels = labels[idx]

    row_order = np.argsort(labels, kind='stable')
    H_sorted = H[row_order]
    sorted_labels = labels[row_order]

    num_classes = int(labels.max()) + 1
    class_counts = np.zeros((H.shape[1], num_classes), dtype=np.float64)
    for cls in range(num_classes):
        class_counts[:, cls] = H.T @ (labels == cls)
    dominant_class = np.where(
        class_counts.sum(axis=1) > 0,
        np.argmax(class_counts, axis=1),
        -1,
    )
    col_order = np.lexsort((np.arange(H.shape[1]), dominant_class))
    H_sorted = H_sorted[:, col_order]

    row_boundaries = [0]
    for pos in range(1, len(sorted_labels)):
        if sorted_labels[pos] != sorted_labels[pos - 1]:
            row_boundaries.append(pos)

    col_boundaries = [0]
    sorted_col_classes = dominant_class[col_order]
    for pos in range(1, len(sorted_col_classes)):
        if sorted_col_classes[pos] != sorted_col_classes[pos - 1]:
            col_boundaries.append(pos)

    return H_sorted, row_order, col_order, sorted_labels, row_boundaries, col_boundaries


def plot_sorted_incidence(H, labels, save_path, title):
    """Render the class-sorted incidence heatmap used for the paper figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f'[topology] matplotlib unavailable, skip {save_path}: {exc}')
        return False

    H_sorted, _, _, _, row_boundaries, col_boundaries = reorder_incidence(H, labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(H_sorted, cmap='Greys', interpolation='nearest',
                   vmin=0, vmax=1)
    for boundary in row_boundaries:
        ax.axhline(boundary - 0.5, color='#d62728', linewidth=1.0)
    for boundary in col_boundaries:
        ax.axvline(boundary - 0.5, color='#d62728', linewidth=1.0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Hyperedge index (sorted by class)')
    ax.set_ylabel('Node index (sorted by class)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return True


def save_topology_snapshot(snapshot, labels, output_dir, dataset_name, run, epoch,
                           tag='snapshot', export_csv=False, export_plot=False,
                           export_class_block=False, params=None,
                           images_subdir='figures'):
    """Persist one topology snapshot plus its quantitative metrics.

    epoch is the 0-based training epoch; filenames use 1-based epoch numbers so
    that "epoch 0" appears as epoch0001 and "epoch 99" as epoch0100.
    """
    if snapshot is None:
        return None

    run_dir = os.path.join(output_dir, dataset_name, f'run{run}')
    os.makedirs(run_dir, exist_ok=True)
    figure_dir = os.path.join(run_dir, images_subdir)
    suffix = f'run{run}_epoch{int(epoch) + 1:04d}_{tag}'

    if params is not None:
        params_path = os.path.join(run_dir, 'run_params.json')
        with open(params_path, 'w', encoding='utf-8') as fh:
            json.dump(params, fh, indent=2)

    saved = {}
    for key in ('H_initial', 'H_evolved', 'H_hard'):
        value = snapshot.get(key)
        if value is not None:
            arr = _as_numpy(value)
            np.save(os.path.join(run_dir, f'{suffix}_{key}.npy'), arr)
            saved[key] = arr

    node_embedding = snapshot.get('node_embedding')
    if node_embedding is not None:
        node_arr = _as_numpy(node_embedding)
        np.save(os.path.join(run_dir, f'{suffix}_node_embedding.npy'), node_arr)
        saved['node_embedding'] = node_arr

    labels_arr = _labels_vector(labels)
    np.save(os.path.join(run_dir, f'{suffix}_labels.npy'), labels_arr)

    metrics = {}
    for key in ('H_initial', 'H_evolved', 'H_hard'):
        if key in saved:
            metrics[key] = compute_topology_metrics(saved[key], labels_arr)

    meta = {
        'dataset': dataset_name,
        'run': run,
        'epoch_0based': int(epoch),
        'epoch_1based': int(epoch) + 1,
        'tag': tag,
        'num_nodes': int(labels_arr.shape[0]),
        'num_classes': int(labels_arr.max()) + 1,
    }
    metrics_path = os.path.join(run_dir, f'{suffix}_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as fh:
        json.dump({'meta': meta, 'metrics': metrics}, fh, indent=2)

    if export_csv:
        for key in ('H_initial', 'H_evolved', 'H_hard'):
            if key in saved:
                reordered, row_order, col_order, sorted_labels, _, _ = reorder_incidence(
                    saved[key], labels_arr)
                csv_path = os.path.join(run_dir, f'{suffix}_{key}_sorted.csv')
                np.savetxt(csv_path, reordered, delimiter=',', fmt='%d')
                np.save(os.path.join(run_dir, f'{suffix}_{key}_row_order.npy'), row_order)
                np.save(os.path.join(run_dir, f'{suffix}_{key}_col_order.npy'), col_order)
                np.save(os.path.join(run_dir, f'{suffix}_{key}_sorted_labels.npy'),
                        sorted_labels)

    if export_plot:
        os.makedirs(figure_dir, exist_ok=True)
        for key in ('H_initial', 'H_evolved', 'H_hard'):
            if key in saved:
                plot_sorted_incidence(
                    saved[key],
                    labels_arr,
                    os.path.join(figure_dir, f'{suffix}_{key}_sorted.png'),
                    title=f'{dataset_name} run {run + 1} epoch {int(epoch) + 1} {key}',
                )

    if export_class_block:
        os.makedirs(figure_dir, exist_ok=True)
        for key in ('H_initial', 'H_evolved', 'H_hard'):
            if key in saved:
                plot_class_block_matrix(
                    saved[key],
                    labels_arr,
                    os.path.join(figure_dir, f'{suffix}_{key}_class_block.png'),
                    title=f'{dataset_name} run {run + 1} epoch {int(epoch) + 1} {key}',
                )

    return metrics


def save_topology_comparison(initial_metrics, final_metrics, output_dir,
                             dataset_name, run, best_epoch,
                             fixed_metrics=None, params=None):
    """Write a compact summary comparing epoch 1 with the evolved topology."""
    run_dir = os.path.join(output_dir, dataset_name, f'run{run}')
    os.makedirs(run_dir, exist_ok=True)

    if params is not None:
        summary_params = dict(params)
        summary_params['best_epoch_0based'] = int(best_epoch)
        summary_params['best_epoch_1based'] = int(best_epoch) + 1
        params_path = os.path.join(run_dir, 'run_params.json')
        with open(params_path, 'w', encoding='utf-8') as fh:
            json.dump(summary_params, fh, indent=2)

    initial_raw = initial_metrics.get('H_initial')
    final_evolved = final_metrics.get('H_evolved')
    final_hard = final_metrics.get('H_hard')

    summary = {
        'dataset': dataset_name,
        'run': run,
        'best_epoch_0based': int(best_epoch),
        'best_epoch_1based': int(best_epoch) + 1,
        'initial_raw': initial_raw,
        'final_evolved': final_evolved,
        'final_hard': final_hard,
    }

    if initial_raw is not None and final_evolved is not None:
        summary['delta_initial_raw_to_final_evolved'] = {
            'homophily': final_evolved['homophily'] - initial_raw['homophily'],
            'homophily_nonempty': (
                final_evolved['homophily_nonempty'] - initial_raw['homophily_nonempty']
            ),
            'effective_hyperedge_ratio': (
                final_evolved['effective_hyperedge_ratio']
                - initial_raw['effective_hyperedge_ratio']
            ),
            'diagonal_block_ratio': (
                final_evolved['diagonal_block_ratio']
                - initial_raw['diagonal_block_ratio']
            ),
            'cross_class_ratio': (
                final_evolved['cross_class_ratio']
                - initial_raw['cross_class_ratio']
            ),
        }

    if fixed_metrics is not None:
        summary['fixed_epoch_evolved'] = fixed_metrics.get('H_evolved')

    summary_path = os.path.join(run_dir, f'run{run}_evolution_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    return summary_path

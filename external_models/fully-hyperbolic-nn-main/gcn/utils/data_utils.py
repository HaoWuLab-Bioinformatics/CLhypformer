"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
import csv
import json

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


def load_data(args, datapath):
    if args.task == "nc":
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data["adj_train"]
        if args.task == "lp":
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                adj, args.val_prop, args.test_prop, args.split_seed
            )
            data["adj_train"] = adj_train
            data["train_edges"], data["train_edges_false"] = train_edges, train_edges_false
            data["val_edges"], data["val_edges_false"] = val_edges, val_edges_false
            data["test_edges"], data["test_edges_false"] = test_edges, test_edges_false
    data["adj_train_norm"], data["features"] = process(
        data["adj_train"], data["features"], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == "airport":
        data["features"] = augment(data["adj_train"], data["features"])
    return data


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    sparse_tensor = sparse_tensor.coalesce()
    return sparse_tensor


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    x, y = sp.triu(sp.csr_matrix(1.0 - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = (
        pos_edges[:n_val],
        pos_edges[n_val : n_test + n_val],
        pos_edges[n_test + n_val :],
    )
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val : n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return (
        adj_train,
        torch.LongTensor(train_edges),
        torch.LongTensor(train_edges_false),
        torch.LongTensor(val_edges),
        torch.LongTensor(val_edges_false),
        torch.LongTensor(test_edges),
        torch.LongTensor(test_edges_false),
    )


def split_data(labels, val_prop, test_prop, seed):
    from sklearn.model_selection import train_test_split

    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    if labels.ndim > 1:
        labels = labels.flatten()
    labels = np.asarray(labels, dtype=np.int64)
    all_idx = np.arange(nb_nodes)
    train_prop = 1.0 - val_prop - test_prop
    train_val_idx, idx_test = train_test_split(
        all_idx, test_size=test_prop, random_state=seed, stratify=labels
    )
    val_relative_prop = val_prop / (train_prop + val_prop)
    idx_train, idx_val = train_test_split(
        train_val_idx, test_size=val_relative_prop, random_state=seed, stratify=labels[train_val_idx]
    )
    return idx_val.tolist(), idx_test.tolist(), idx_train.tolist()


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ["cora", "pubmed", "citeseer"]:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == "disease_lp":
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == "airport":
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError("Dataset {} is not supported.".format(dataset))
    data = {"adj_train": adj, "features": features}
    return data


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ["cora", "pubmed", "citeseer"]:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == "disease_nc":
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == "airport":
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset == "email-Eu-core":
            adj, features, labels = load_email_eu_core(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.15
        elif dataset.startswith("twitch-"):
            adj, features, labels = load_twitch_dataset(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.15
        else:
            raise FileNotFoundError("Dataset {} is not supported.".format(dataset))

        print(f"\n=== Dataset: {dataset} ===")
        print(f"Number of nodes: {adj.shape[0]}")
        print(f"Number of edges: {int(adj.nnz // 2)}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        print(f"Label distribution:")
        for label in np.unique(labels):
            count = np.sum(labels == label)
            print(f"  Class {label}: {count} samples ({count/len(labels)*100:.2f}%)")

        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

        print(f"Train size: {len(idx_train)}, Val size: {len(idx_val)}, Test size: {len(idx_test)}")
        print("=" * 40 + "\n")

    labels = torch.LongTensor(labels)
    data = {
        "adj_train": adj,
        "features": features,
        "labels": labels,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
    }
    return data


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_reorder), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_reorder), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + min(1000, len(labels) - len(y) - len(idx_test)))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])

    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    if labels.dtype == np.float64 or labels.dtype == np.float32:
        labels = np.round(labels).astype(np.int64)
    elif labels.ndim > 1:
        labels = np.argmax(labels, axis=1)
    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_idx[label] for label in labels])
    num_nodes = labels.shape[0]

    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), "r") as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(",")
        edges.append((int(n1), int(n2)))

    adj = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj = adj + np.eye(num_nodes)
    adj = sp.csr_matrix(adj)

    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    else:
        features = sp.eye(num_nodes)
    return adj, features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + ".p"), "rb"))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]["feat"] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


def load_email_eu_core(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    edges_file = os.path.join(data_path, "email-Eu-core.txt")
    labels_file = os.path.join(data_path, "email-Eu-core-department-labels.txt")
    if os.path.exists(edges_file):
        with open(edges_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        n1, n2 = int(parts[0]), int(parts[1])
                        if n1 not in object_to_idx:
                            object_to_idx[n1] = idx_counter
                            idx_counter += 1
                        if n2 not in object_to_idx:
                            object_to_idx[n2] = idx_counter
                            idx_counter += 1
                        edges.append((object_to_idx[n1], object_to_idx[n2]))
                    except ValueError:
                        continue
    num_nodes = len(object_to_idx)
    adj = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    if use_feats:
        features = sp.eye(num_nodes)
    else:
        features = sp.eye(num_nodes)
    if os.path.exists(labels_file):
        labels_dict = {}
        max_node_id = 0
        with open(labels_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        node_id = int(parts[0])
                        label = int(parts[1])
                        labels_dict[node_id] = label
                        if node_id > max_node_id:
                            max_node_id = node_id
                    except ValueError:
                        continue
        labels = np.zeros(max_node_id + 1, dtype=np.int64)
        for nid, lab in labels_dict.items():
            labels[nid] = lab
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n=== Email-Eu-Core original label distribution ===")
        print(f"Total nodes: {len(labels)}")
        print(f"Original classes: {len(unique_labels)}")
        print(f"Class distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples ({count/len(labels)*100:.2f}%)")
        min_samples_per_class = 3
        valid_labels_mask = np.zeros(len(labels), dtype=bool)
        for label, count in zip(unique_labels, counts):
            if count >= min_samples_per_class:
                valid_labels_mask |= labels == label
        filtered_labels = labels[valid_labels_mask]
        filtered_node_ids = np.where(valid_labels_mask)[0]
        unique_valid_labels = np.unique(filtered_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_valid_labels)}
        remapped_labels = np.array([label_to_idx[label] for label in filtered_labels])
        valid_indices_map = {old_id: new_id for new_id, old_id in enumerate(filtered_node_ids)}
        new_num_nodes = len(filtered_node_ids)
        new_adj_rows = []
        new_adj_cols = []
        for i, j in edges:
            if i in valid_indices_map and j in valid_indices_map:
                new_i = valid_indices_map[i]
                new_j = valid_indices_map[j]
                new_adj_rows.extend([new_i, new_j])
                new_adj_cols.extend([new_j, new_i])
        new_adj_data = np.ones(len(new_adj_rows))
        adj = sp.csr_matrix(
            (new_adj_data, (new_adj_rows, new_adj_cols)), shape=(new_num_nodes, new_num_nodes)
        )
        features = sp.eye(new_num_nodes)
        labels = remapped_labels
        print(f"\n=== Filtered label distribution ===")
        print(f"Filtered nodes: {len(labels)}")
        print(f"Filtered classes: {len(np.unique(labels))}")
        final_unique, final_counts = np.unique(labels, return_counts=True)
        for label, count in zip(final_unique, final_counts):
            print(f"  Class {label}: {count} samples ({count/len(labels)*100:.2f}%)")
        print("=" * 40 + "\n")
    else:
        raise FileNotFoundError("Labels file not found for {}".format(dataset_str))
    return sp.csr_matrix(adj), features, labels


def load_twitch_dataset(dataset_str, use_feats, data_path):
    lang_code = dataset_str.split("-")[1] if "-" in dataset_str else "PTBR"
    edges_file = os.path.join(data_path, f"musae_{lang_code}_edges.csv")
    labels_file = os.path.join(data_path, f"musae_{lang_code}_target.csv")
    feats_file = os.path.join(data_path, f"musae_{lang_code}_features.json")
    print(f"[Debug] Twitch-{lang_code} labels file path: {os.path.abspath(labels_file)}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found at: {labels_file} (Dataset: {dataset_str})")
    if not os.path.exists(edges_file):
        raise FileNotFoundError(f"Edges file not found at: {edges_file}")

    labels_dict = {}
    max_node_id = 0
    with open(labels_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_col = header.index("id")
        mature_col = header.index("mature")
        new_id_col = header.index("new_id")
        for row in reader:
            if not row:
                continue
            try:
                new_id = int(row[new_id_col])
                mature_str = row[mature_col].strip().lower()
                label = 1 if mature_str in ["true", "1"] else 0
                labels_dict[new_id] = label
                if new_id > max_node_id:
                    max_node_id = new_id
            except (ValueError, IndexError):
                continue
    num_nodes = max_node_id + 1
    labels = np.zeros(num_nodes, dtype=np.int64)
    for nid, lab in labels_dict.items():
        labels[nid] = lab
    row_indices = []
    col_indices = []
    with open(edges_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == 2:
                try:
                    n1, n2 = int(row[0]), int(row[1])
                    row_indices.extend([n1, n2])
                    col_indices.extend([n2, n1])
                except ValueError:
                    continue
    data = np.ones(len(row_indices))
    adj = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    adj.data = np.ones_like(adj.data)

    if use_feats:
        if os.path.exists(feats_file):
            with open(feats_file, "r", encoding="utf-8") as f:
                features_dict = json.load(f)
            max_feat_idx = 0
            for feat_list in features_dict.values():
                if feat_list:
                    max_feat_idx = max(max_feat_idx, max(feat_list))
            num_features = max_feat_idx + 1
            rows_f = []
            cols_f = []
            for node_str, feat_list in features_dict.items():
                try:
                    nid = int(node_str)
                    if nid < num_nodes:
                        for f_idx in feat_list:
                            rows_f.append(nid)
                            cols_f.append(f_idx)
                except ValueError:
                    continue
            data_f = np.ones(len(rows_f))
            features = sp.csr_matrix((data_f, (rows_f, cols_f)), shape=(num_nodes, num_features))
            from sklearn.preprocessing import normalize
            features = normalize(features, norm="l2", axis=1)
            features = features * 0.5
        else:
            features = sp.eye(num_nodes)

    print("\n" + "=" * 50)
    print("[Debug] Twitch-PTBR dataset loading diagnostics")
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f" -> Global label distribution: {label_dist}")
    if len(counts) > 1:
        imbalance_ratio = max(counts) / min(counts)
        print(f" -> Class imbalance ratio: {imbalance_ratio:.2f} : 1")
    degrees = np.array(adj.sum(axis=1)).flatten()
    active_nodes = len(np.unique(adj.nonzero()[0]))
    isolated_nodes = num_nodes - active_nodes
    print(f" -> Graph structure: {num_nodes} nodes, {adj.nnz // 2} edges, {isolated_nodes} isolated nodes")
    print(f" -> Degree stats: mean={degrees.mean():.2f}, max={degrees.max()}, median={np.median(degrees)}")
    if sp.issparse(features):
        active_feat_nodes = len(np.unique(features.nonzero()[0]))
        nnz = features.nnz
    else:
        active_feat_nodes = np.count_nonzero(features.sum(axis=1))
        nnz = np.count_nonzero(features)
    zero_feat_nodes = num_nodes - active_feat_nodes
    print(f" -> Feature space: dim={features.shape[1]}, nonzeros={nnz}, missing features={zero_feat_nodes}")
    print("=" * 50 + "\n")
    return adj, features, labels


def edge_mask_adj(adj, rate, seed=None):
    """
    Randomly mask (remove) a proportion of edges from the adjacency matrix.
    Only non-self-loop edges are considered for removal. The resulting matrix
    remains symmetric.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Input adjacency matrix (symmetric)
    rate : float
        Proportion of edges to remove, 0 <= rate < 1
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    adj_masked : scipy.sparse.csr_matrix
        Adjacency matrix with a fraction of edges removed
    """
    if rate <= 0.0:
        return adj
    if rate >= 1.0:
        raise ValueError("Edge masking rate must be strictly less than 1")

    if seed is not None:
        np.random.seed(seed)

    # Convert to LIL for efficient row/col indexing assignment
    adj = adj.copy().tolil()

    # Get upper-triangular non-diagonal edge indices (each undirected edge once)
    rows, cols = sp.triu(adj, k=1).nonzero()
    n_edges = len(rows)
    n_remove = int(n_edges * rate)

    if n_remove == 0:
        return adj.tocsr()

    # Randomly choose which edges to remove
    remove_idx = np.random.choice(n_edges, n_remove, replace=False)
    remove_rows = rows[remove_idx]
    remove_cols = cols[remove_idx]

    # Remove in both directions to preserve symmetry
    adj[remove_rows, remove_cols] = 0
    adj[remove_cols, remove_rows] = 0

    return adj.tocsr()

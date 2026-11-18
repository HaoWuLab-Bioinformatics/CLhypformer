"""Generic-class cohort evaluator derived from frozen gse_inductive_eval.py."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from sklearn.neighbors import kneighbors_graph

from gse_shared_eval import load_original, original_args, set_seed, sha256, split_sha


def _gradient_diagnostics(task_loss, cl_loss, model):
    params = tuple(p for p in (
        tuple(model.adv_module.encoder_l1.parameters())
        + tuple(model.adv_module.encoder_l2.parameters())) if p.requires_grad)
    task = torch.autograd.grad(task_loss, params, retain_graph=True, allow_unused=True)
    contrast = torch.autograd.grad(cl_loss, params, retain_graph=True, allow_unused=True)
    task_sq = sum((g.detach() ** 2).sum() for g in task if g is not None)
    cl_sq = sum((g.detach() ** 2).sum() for g in contrast if g is not None)
    dot = sum((a.detach() * b.detach()).sum()
              for a, b in zip(task, contrast) if a is not None and b is not None)
    task_norm = float(task_sq.sqrt()) if isinstance(task_sq, torch.Tensor) else 0.0
    cl_norm = float(cl_sq.sqrt()) if isinstance(cl_sq, torch.Tensor) else 0.0
    return {"task_grad_norm": task_norm, "cl_grad_norm": cl_norm,
            "cl_to_task_grad_ratio": cl_norm / max(task_norm, 1e-12),
            "grad_cosine": float(dot) / max(task_norm * cl_norm, 1e-12)}


def _make_graph(features, label, minor, patients, global_idx, device, original):
    if len(features) < 2:
        raise ValueError("Each patient graph must contain at least two cells")
    neighbors = min(10, len(features) - 1)
    adj = kneighbors_graph(features, n_neighbors=neighbors, include_self=False)
    adj = adj.maximum(adj.T).tocoo()
    edge = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64)).to(device)
    classes, minor_code = np.unique(minor, return_inverse=True)
    if np.all(label < 0):
        minor_code = np.full(len(label), -1, dtype=np.int64)
    _, patient_code = np.unique(patients, return_inverse=True)
    graph = {"node_feat": torch.from_numpy(features.astype(np.float32)).to(device),
             "edge_index": (edge,) if original else edge,
             "edge_feat": None, "num_nodes": len(features)}
    return SimpleNamespace(
        graph=graph, label=torch.from_numpy(label.astype(np.int64))[:, None].to(device),
        cl_fine_labels=torch.from_numpy(minor_code).long().to(device),
        patient_codes=torch.from_numpy(patient_code).long().to(device),
        global_idx=global_idx, minor_classes=classes)


def _predict(model, datasets):
    model.eval()
    indices, logits = [], []
    with torch.no_grad():
        for dataset in datasets:
            result = model(dataset)
            indices.append(dataset.global_idx)
            logits.append(result.cpu())
    return np.concatenate(indices), torch.cat(logits).numpy()


def run(cli):
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cli.npz) as raw:
        data = {key: raw[key].copy() for key in raw.files}
    device = torch.device(f"cuda:{cli.device}" if torch.cuda.is_available()
                          and cli.device >= 0 else "cpu")
    set_seed(cli.seed)
    original = cli.model == "original"
    train_idx = data["train_idx"]
    valid_idx = data["valid_idx"]
    test_idx = data["test_idx"]
    # Validation/test labels are masked inside model inputs; labels remain in
    # the external data array solely for scoring after forward inference.
    train = _make_graph(data["features"][train_idx], data["labels"][train_idx],
                        data["celltype_minor"][train_idx],
                        data["patient_ids"][train_idx], train_idx, device, original)
    if cli.graph_policy == "patient_mnn":
        from gse_patient_modules import patient_aware_edges
        edges, _, diagnostics = patient_aware_edges(
            data["features"][train_idx], data["patient_ids"][train_idx])
        graph_edges = torch.from_numpy(edges).long().to(device)
        train.graph["edge_index"] = (graph_edges,) if original else graph_edges
        (cli.output_dir / "graph_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2))
    elif cli.graph_policy == "patient_local":
        from gse_patient_state_graph import patient_local_edges
        edges, diagnostics = patient_local_edges(
            data["features"][train_idx], data["patient_ids"][train_idx])
        graph_edges = torch.from_numpy(edges).long().to(device)
        train.graph["edge_index"] = (graph_edges,) if original else graph_edges
        (cli.output_dir / "graph_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2))
    def patients_as_graphs(indices):
        output = []
        for patient in np.unique(data["patient_ids"][indices]):
            selected = indices[data["patient_ids"][indices] == patient]
            item = _make_graph(
                data["features"][selected], np.full(len(selected), -1),
                data["celltype_minor"][selected], data["patient_ids"][selected],
                selected, device, original)
            if cli.graph_policy == "patient_local":
                from gse_patient_state_graph import patient_local_edges
                local_edges, _ = patient_local_edges(
                    data["features"][selected], data["patient_ids"][selected])
                graph_edges = torch.from_numpy(local_edges).long().to(device)
                item.graph["edge_index"] = ((graph_edges,) if original
                                            else graph_edges)
            output.append(item)
        return output
    validation = patients_as_graphs(valid_idx)
    testing = patients_as_graphs(test_idx)
    args = original_args(data["features"].shape[1], len(data["class_names"]))
    args.device = device
    if original:
        HypFormer, Optimizer = load_original()
        source = Path(__file__).resolve().parent.parent / "hyperbolic-transformer-master-ori/medium/hypformer.py"
    else:
        from hypformer import HypFormer
        from manifolds.hyp_layer import Optimizer
        source = Path(__file__).resolve().parent / "hypformer.py"
        args.use_cl = bool(cli.use_cl)
        args.use_adaptive_curvature = False
        args.cl_mode = cli.cl_mode
        args.hyperedge_policy = cli.hyperedge_policy
        args.prototype_temperature = cli.prototype_temperature
        args.cl_weight = cli.cl_weight
        args.cl_fusion_weight = 0.0
        args.cl_aux_head = cli.aux_head
        args.cl_consistency = cli.consistency
        args.cl_consistency_weight = 0.2
        args.cl_fine_labels = cli.fine_labels
        args.cl_cross_patient = cli.cross_patient
        args.cl_pair_policy = cli.pair_policy
        args.cl_mask_same_major = cli.mask_same_major
        args.cl_patient_balance = cli.patient_balance
        args.cl_knn = 5
        args.cl_drop_rate = 0.2
        args.cl_tau = 0.7
        args.knn_metric = "hyp"
        args.drop_strategy = "random"
        args.cl_space = "hyp"
        args.graph_build_method = "knn"
        args.graph_source = "learned"
        args.cl_feature_noise = 0.05
        args.save_topology = 0
    model = HypFormer(args).to(device)
    train_labels = train.label.squeeze(1)
    if not original and cli.use_cl and not cli.no_pretrain:
        from geoopt.optim import RiemannianAdam
        pre_optimizer = RiemannianAdam(model.adv_module.parameters(), lr=args.lr)
        all_train = torch.arange(len(train_idx), device=device)
        for epoch in range(cli.pretrain_epochs):
            model.train()
            pre_optimizer.zero_grad()
            _, _, cl_loss, _ = model.adv_module(
                train.graph["node_feat"],
                labels=train.cl_fine_labels if cli.fine_labels else train_labels,
                train_idx=all_train, objective="supcon",
                group_ids=train.patient_codes if cli.cross_patient else None,
                cross_group=bool(cli.cross_patient),
                coarse_labels=train_labels,
                mask_same_major=bool(cli.mask_same_major),
                pair_policy=cli.pair_policy,
                patient_balance=bool(cli.patient_balance))
            cl_loss.backward()
            pre_optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"pretrain={epoch + 1} loss={cl_loss.item():.5f}", flush=True)
    optimizer = Optimizer(model, args)
    best_val, best_epoch, best_state, patience = -1., -1, None, 0
    best_score = -1.0
    best_val_metrics = None
    for epoch in range(cli.epochs):
        model.train()
        optimizer.zero_grad()
        if not original and cli.use_cl:
            if epoch < cli.cl_warmup_epochs:
                model.cl_weight = 0.0
            elif cli.cl_ramp_epochs:
                progress = (epoch - cli.cl_warmup_epochs + 1) / cli.cl_ramp_epochs
                model.cl_weight = cli.cl_weight * min(1.0, progress)
            else:
                model.cl_weight = cli.cl_weight
            output, _, _, cl_loss = model(train, train_idx=torch.arange(
                len(train_idx), device=device))
            task_loss = F.cross_entropy(output, train_labels)
            loss = task_loss + cl_loss
            if cli.grad_diagnostics and (epoch == 0 or (epoch + 1) % 10 == 0):
                if cli.cl_mode == "patient_prototype":
                    from gse_patient_modules import prototype_gradient_diagnostics
                    diagnostic = prototype_gradient_diagnostics(task_loss, cl_loss,
                                                                 model)
                    diagnostic["prototype_stats"] = model.last_prototype_stats
                else:
                    diagnostic = _gradient_diagnostics(task_loss, cl_loss, model)
                diagnostic.update({"epoch": epoch, "cl_weight": model.cl_weight,
                                   "task_loss": float(task_loss.detach()),
                                   "cl_branch_loss": float(cl_loss.detach())})
                with (cli.output_dir / "cl_gradients.jsonl").open("a") as stream:
                    stream.write(json.dumps(diagnostic) + "\n")
        else:
            output = model(train)
            loss = F.cross_entropy(output, train_labels)
        if not original and cli.use_cl and cli.prototype_pcgrad:
            from gse_patient_modules import apply_prototype_pcgrad
            pcgrad_status = apply_prototype_pcgrad(task_loss, cl_loss, model)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print("Prototype PCGrad: " + json.dumps(pcgrad_status), flush=True)
        else:
            loss.backward()
        optimizer.step()
        val_positions, val_logits = _predict(model, validation)
        val_pred = val_logits.argmax(axis=1)
        val_true = data["labels"][val_positions]
        val_acc = float(np.mean(val_pred == val_true))
        val_macro = float(f1_score(val_true, val_pred,
                                   labels=np.unique(val_true), average="macro",
                                   zero_division=0))
        score = val_acc if cli.checkpoint_metric == "accuracy" else val_macro
        if epoch + 1 >= cli.min_checkpoint_epoch and score > best_score:
            best_score = score
            best_val, best_epoch = float(val_acc), epoch
            best_val_metrics = {
                "macro_f1": val_macro,
                "class_recall": recall_score(val_true, val_pred,
                                              labels=np.arange(len(data["class_names"])), average=None,
                                              zero_division=0).tolist(),
                "class_support": np.bincount(val_true, minlength=len(data["class_names"])).tolist(),
            }
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        elif epoch + 1 >= cli.min_checkpoint_epoch:
            patience += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} train_loss={loss.item():.5f} "
                  f"val_acc={val_acc:.5f} val_macro_f1={val_macro:.5f}",
                  flush=True)
        if patience >= cli.patience:
            break
    model.load_state_dict(best_state)
    validation_idx, validation_logits = _predict(model, validation)
    if not np.array_equal(np.sort(validation_idx), np.sort(valid_idx)):
        raise AssertionError("Missing validation patient cells")
    validation_prob = torch.from_numpy(validation_logits).softmax(-1).numpy()
    np.savez_compressed(cli.output_dir / "validation_predictions.npz",
                        valid_idx=validation_idx,
                        true=data["labels"][validation_idx],
                        pred=validation_prob.argmax(-1), prob=validation_prob)
    idx, logits = _predict(model, testing)
    if not np.array_equal(np.sort(idx), np.sort(test_idx)):
        raise AssertionError("Missing test patient cells")
    probs = torch.from_numpy(logits).softmax(-1).numpy()
    pred = probs.argmax(-1)
    truth = data["labels"][idx]
    np.savez_compressed(cli.output_dir / "predictions.npz", test_idx=idx,
                        true=truth, pred=pred, prob=probs)
    torch.save({"state_dict": best_state, "best_epoch": best_epoch,
                "best_validation_accuracy": best_val,
                "checkpoint_metric": cli.checkpoint_metric,
                "selection_score": best_score,
                "best_validation_macro_f1": best_val_metrics["macro_f1"],
                "validation_class_recall": best_val_metrics["class_recall"],
                "validation_class_support": best_val_metrics["class_support"]},
               cli.output_dir / "checkpoint.pt")
    result = {
        "protocol": "inductive_patient_graphs", "model": cli.model,
        "use_cl": cli.use_cl, "seed": cli.seed,
        "fine_labels": cli.fine_labels, "cross_patient": cli.cross_patient,
        "pair_policy": cli.pair_policy, "mask_same_major": cli.mask_same_major,
        "patient_balance": cli.patient_balance, "cl_weight": cli.cl_weight,
        "cl_warmup_epochs": cli.cl_warmup_epochs,
        "cl_ramp_epochs": cli.cl_ramp_epochs,
        "no_pretrain": cli.no_pretrain,
        "cl_mode": cli.cl_mode,
        "hyperedge_policy": cli.hyperedge_policy,
        "prototype_temperature": cli.prototype_temperature,
        "prototype_pcgrad": cli.prototype_pcgrad,
        "graph_policy": cli.graph_policy,
        "split_sha256": split_sha(data), "data_sha256": sha256(cli.npz),
        "source_sha256": sha256(source), "evaluator_sha256": sha256(Path(__file__)),
        "best_epoch": best_epoch, "best_validation_accuracy": best_val * 100,
        "checkpoint_metric": cli.checkpoint_metric,
        "min_checkpoint_epoch": cli.min_checkpoint_epoch,
        "best_validation_macro_f1": best_val_metrics["macro_f1"] * 100,
        "validation_class_recall": best_val_metrics["class_recall"],
        "validation_class_support": best_val_metrics["class_support"],
        "test_accuracy": float(np.mean(pred == truth) * 100),
        "test_macro_f1": float(f1_score(truth, pred, labels=np.arange(len(data["class_names"])),
                                         average="macro", zero_division=0)),
        "class_recall": recall_score(truth, pred, labels=np.arange(len(data["class_names"])),
                                     average=None, zero_division=0).tolist(),
        "class_support": np.bincount(truth, minlength=len(data["class_names"])).tolist(),
    }
    (cli.output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("original", "modified"), required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--use-cl", type=int, choices=(0, 1), default=0)
    parser.add_argument("--fine-labels", type=int, choices=(0, 1), default=1)
    parser.add_argument("--cross-patient", type=int, choices=(0, 1), default=1)
    parser.add_argument("--pair-policy", choices=("diag_fixed", "legacy"), default="diag_fixed")
    parser.add_argument("--mask-same-major", type=int, choices=(0, 1), default=0)
    parser.add_argument("--patient-balance", type=int, choices=(0, 1), default=0)
    parser.add_argument("--cl-weight", type=float, default=0.2)
    parser.add_argument("--cl-mode", choices=("shared_supcon", "patient_prototype"),
                        default="shared_supcon")
    parser.add_argument("--hyperedge-policy",
                        choices=("global_learned", "patient_state"),
                        default="global_learned")
    parser.add_argument("--prototype-temperature", type=float, default=0.2)
    parser.add_argument("--prototype-pcgrad", type=int, choices=(0, 1), default=0)
    parser.add_argument("--graph-policy",
                        choices=("global_knn", "patient_mnn", "patient_local"),
                        default="global_knn")
    parser.add_argument("--cl-warmup-epochs", type=int, default=0)
    parser.add_argument("--cl-ramp-epochs", type=int, default=0)
    parser.add_argument("--aux-head", type=int, choices=(0, 1), default=1)
    parser.add_argument("--consistency", type=int, choices=(0, 1), default=1)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--grad-diagnostics", type=int, choices=(0, 1), default=0)
    parser.add_argument("--checkpoint-metric", choices=("accuracy", "macro_f1"),
                        default="accuracy")
    parser.add_argument("--min-checkpoint-epoch", type=int, default=1)
    args = parser.parse_args()
    if args.mask_same_major and not args.fine_labels:
        parser.error("same-major masking requires fine labels")
    if args.patient_balance and not args.cross_patient:
        parser.error("patient balancing requires cross-patient positives")
    if args.cl_warmup_epochs < 0 or args.cl_ramp_epochs < 0:
        parser.error("CL warmup and ramp lengths must be nonnegative")
    if args.min_checkpoint_epoch < 1 or args.min_checkpoint_epoch > args.epochs:
        parser.error("min-checkpoint-epoch must be between 1 and epochs")
    if args.cl_mode == "patient_prototype" and not args.no_pretrain:
        parser.error("patient-prototype mode requires --no-pretrain")
    if args.prototype_temperature <= 0:
        parser.error("prototype-temperature must be positive")
    if args.prototype_pcgrad and args.cl_mode != "patient_prototype":
        parser.error("prototype-pcgrad requires patient-prototype mode")
    if args.hyperedge_policy == "patient_state" and (
            args.model != "modified" or not args.use_cl or
            args.cl_mode != "patient_prototype"):
        parser.error("patient-state hyperedges require modified patient-prototype fusion")
    run(args)


if __name__ == "__main__":
    main()

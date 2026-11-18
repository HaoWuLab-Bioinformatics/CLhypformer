"""Common GSE176078 entry for original and modified HypFormer evaluation.

Both paths consume the same prepared NPZ. The original implementation is
loaded from its own source tree without editing that tree. Checkpoints are
chosen exclusively by the configured validation metric.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score


MEDIUM = Path(__file__).resolve().parent
ORIGINAL_MEDIUM = MEDIUM.parent / "hyperbolic-transformer-master-ori" / "medium"
CLASSES = 9


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sha(data):
    joined = np.concatenate([data[k].astype(np.int64) for k in
                             ("train_idx", "valid_idx", "test_idx")])
    return hashlib.sha256(joined.tobytes()).hexdigest()[:16]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def original_args(input_dim, n_classes):
    """Use the same task hyperparameters as bio_runner.py."""
    return SimpleNamespace(
        dataset="gse176078_brca", method="hypformer",
        in_channels=input_dim, hidden_channels=256, out_channels=n_classes,
        use_graph=1, graph_weight=0.8, decoder_type="euc",
        gnn_num_layers=3, gnn_dropout=0.5, gnn_use_bn=0,
        gnn_use_residual=1, gnn_use_weight=0, gnn_use_init=0, gnn_use_act=1,
        trans_num_heads=1, trans_heads_concat=0, trans_use_weight=1,
        trans_use_bn=0, trans_use_residual=1, trans_use_act=0,
        trans_num_layers=1, trans_dropout=0.5,
        k_in=1.0, k_out=3.0, power_k=1.0, attention_type="linear_focused",
        add_positional_encoding=1, k_neighbors=5, use_dynamic_hyp_knn=False,
        lr=0.005, hyp_lr=0.01, optimizer_type="adam",
        hyp_optimizer_type="radam", weight_decay=0.005,
        hyp_weight_decay=0.005,
    )


def load_original():
    if not (ORIGINAL_MEDIUM / "hypformer.py").is_file():
        raise FileNotFoundError(ORIGINAL_MEDIUM / "hypformer.py")
    sys.path.insert(0, str(ORIGINAL_MEDIUM))
    spec = importlib.util.spec_from_file_location(
        "original_hypformer", ORIGINAL_MEDIUM / "hypformer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from manifolds.hyp_layer import Optimizer
    return module.HypFormer, Optimizer


def original_run(cli, data, output_dir):
    HypFormer, Optimizer = load_original()
    device = torch.device(f"cuda:{cli.device}" if torch.cuda.is_available()
                          and cli.device >= 0 else "cpu")
    set_seed(cli.seed)
    graph = {
        "node_feat": torch.from_numpy(data["features"].copy()).to(device),
        # Original forward() accesses edge_index[0].
        "edge_index": (torch.from_numpy(data["edge_index"].copy()).long().to(device),),
        "num_nodes": len(data["labels"]), "edge_feat": None,
    }
    labels = torch.from_numpy(data["labels"].copy()).long().to(device)
    dataset = SimpleNamespace(graph=graph, label=labels[:, None])
    indices = {key: torch.from_numpy(data[f"{key}_idx"].copy()).long().to(device)
               for key in ("train", "valid", "test")}
    args = original_args(graph["node_feat"].shape[1], int(labels.max().item()) + 1)
    model = HypFormer(args).to(device)
    optimizer = Optimizer(model, args)
    best_val = -1.0
    best_score = -1.0
    best_val_metrics = None
    best_state = None
    best_epoch = -1
    wait = 0
    for epoch in range(cli.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(dataset)
        loss = F.cross_entropy(logits[indices["train"]], labels[indices["train"]])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(dataset)
            val_pred = logits[indices["valid"]].argmax(-1).cpu().numpy()
            val_true = labels[indices["valid"]].cpu().numpy()
            val_acc = float(np.mean(val_pred == val_true))
            val_macro = float(f1_score(val_true, val_pred,
                                       labels=np.unique(val_true), average="macro",
                                       zero_division=0))
        score = val_acc if cli.checkpoint_metric == "accuracy" else val_macro
        if epoch + 1 >= cli.min_checkpoint_epoch and score > best_score:
            best_score = score
            best_val, best_epoch = val_acc, epoch
            best_val_metrics = {
                "macro_f1": val_macro,
                "class_recall": recall_score(val_true, val_pred,
                                              labels=np.arange(CLASSES), average=None,
                                              zero_division=0).tolist(),
                "class_support": np.bincount(val_true, minlength=CLASSES).tolist(),
            }
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        elif epoch + 1 >= cli.min_checkpoint_epoch:
            wait += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} train_loss={loss.item():.6f} val_acc={val_acc:.6f}",
                  flush=True)
        if wait >= cli.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(dataset)
        test_idx = indices["test"]
        probs = logits[test_idx].softmax(-1).cpu().numpy()
        predicted = probs.argmax(-1)
        truth = labels[test_idx].cpu().numpy()
        # Pre-classifier transformer and graph representations for later analysis.
        trans_hyp = model.trans_conv(graph["node_feat"])
        trans_euc = model.manifold_out.logmap0(trans_hyp)[..., 1:]
        graph_euc = model.graph_conv(graph["node_feat"], graph["edge_index"][0])
        embedding = torch.cat((trans_euc, graph_euc), dim=-1).cpu().numpy()
    pred_path = output_dir / "predictions.npz"
    np.savez_compressed(pred_path, test_idx=test_idx.cpu().numpy(),
                        true=truth, pred=predicted, prob=probs)
    np.savez_compressed(output_dir / "embeddings.npz", node_idx=np.arange(len(labels)),
                        embedding=embedding, patient_ids=data["patient_ids"],
                        major_labels=data["labels"], minor_labels=data["celltype_minor"])
    torch.save({"state_dict": best_state, "best_epoch": best_epoch,
                "best_validation_accuracy": best_val,
                "checkpoint_metric": cli.checkpoint_metric,
                "selection_score": best_score,
                "best_validation_macro_f1": best_val_metrics["macro_f1"],
                "validation_class_recall": best_val_metrics["class_recall"],
                "validation_class_support": best_val_metrics["class_support"]},
               output_dir / "checkpoint.pt")
    result = {
        "model": "original_source", "seed": cli.seed,
        "split_sha256": split_sha(data), "data_sha256": sha256(cli.npz),
        "source_sha256": sha256(ORIGINAL_MEDIUM / "hypformer.py"),
        "evaluator_sha256": sha256(Path(__file__)),
        "best_epoch": best_epoch, "best_validation_accuracy": best_val * 100,
        "checkpoint_metric": cli.checkpoint_metric,
        "min_checkpoint_epoch": cli.min_checkpoint_epoch,
        "best_validation_macro_f1": best_val_metrics["macro_f1"] * 100,
        "validation_class_recall": best_val_metrics["class_recall"],
        "validation_class_support": best_val_metrics["class_support"],
        "test_accuracy": float((truth == predicted).mean() * 100),
        "test_macro_f1": float(f1_score(truth, predicted,
                                         labels=np.arange(CLASSES),
                                         average="macro", zero_division=0)),
        "class_recall": recall_score(truth, predicted, labels=np.arange(CLASSES),
                                     average=None, zero_division=0).tolist(),
        "class_support": np.bincount(truth, minlength=CLASSES).tolist(),
        "prediction_file": str(pred_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def modified_run(cli, output_dir):
    command = [sys.executable, str(MEDIUM / "main.py"),
               "--dataset", "gse176078_brca", "--method", "hypformer",
               "--data_dir", str(cli.npz.parent.parent), "--seed", str(cli.seed),
               "--epochs", str(cli.epochs), "--runs", "1",
               "--patience", str(cli.patience), "--device", str(cli.device),
               "--rand_split", "1", "--rand_split_class", "1",
               "--use_cl", str(cli.use_cl),
               "--prediction_file", str(output_dir / "predictions.npz"),
               "--robustness_file", str(output_dir / "robustness.json"),
               "--checkpoint_file", str(output_dir / "checkpoint.pt"),
               "--embedding_file", str(output_dir / "embeddings.npz"),
               "--pretrain_epochs", "20", "--train_prop", "0.7",
               "--valid_prop", "0.15", "--save_topology", "0",
               "--display_step", "10", "--lr", "0.005",
               "--weight_decay", "0.005", "--hidden_channels", "256",
               "--use_graph", "1", "--gnn_dropout", "0.5",
               "--gnn_use_bn", "0", "--gnn_num_layers", "3",
               "--gnn_use_init", "0", "--trans_num_layers", "1",
               "--trans_use_residual", "1", "--trans_use_bn", "0",
               "--graph_weight", "0.8", "--trans_dropout", "0.5",
               "--power_k", "1.0", "--decoder_type", "euc",
               "--k_in", "1.0", "--k_out", "3.0", "--knn_num", "5",
               "--attention_type", "linear_focused", "--optimizer_type", "adam",
               "--hyp_optimizer_type", "radam", "--cl_weight", str(cli.cl_weight),
               "--cl_mode", cli.cl_mode,
               "--hyperedge_policy", cli.hyperedge_policy,
               "--prototype_temperature", str(cli.prototype_temperature),
               "--prototype_pcgrad", str(cli.prototype_pcgrad),
               "--cl_aux_weight", "0.1",
               "--cl_fusion_weight", "0.0", "--cl_tau", "0.7",
               "--cl_knn", "5", "--cl_drop_rate", "0.2",
               "--cl_feature_noise", "0.05", "--cl_aux_head", str(cli.aux_head),
               "--cl_consistency", str(cli.consistency),
               "--cl_consistency_weight", "0.2",
               "--cl_fine_labels", str(cli.fine_labels),
               "--cl_cross_patient", str(cli.cross_patient),
               "--cl_pair_policy", cli.pair_policy,
               "--cl_mask_same_major", str(cli.mask_same_major),
               "--cl_patient_balance", str(cli.patient_balance),
               "--cl_warmup_epochs", str(cli.cl_warmup_epochs),
               "--cl_ramp_epochs", str(cli.cl_ramp_epochs),
               "--cl_grad_diagnostics", str(cli.grad_diagnostics),
               "--cl_grad_file", str(output_dir / "cl_gradients.jsonl"),
               "--checkpoint_metric", cli.checkpoint_metric,
               "--min_checkpoint_epoch", str(cli.min_checkpoint_epoch),
               "--drop_strategy", "random", "--topology_label_weight", "0.1"]
    if cli.no_pretrain:
        command.append("--no-pretrain")
    with (output_dir / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=MEDIUM, stdout=log, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, "PYTHONHASHSEED": str(cli.seed)})
    with np.load(output_dir / "predictions.npz") as pred:
        truth, predicted = pred["true"], pred["pred"]
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu",
                            weights_only=False)
    result = {
        "model": "modified", "seed": cli.seed, "split_sha256": split_sha(np.load(cli.npz)),
        "data_sha256": sha256(cli.npz), "source_sha256": sha256(MEDIUM / "hypformer.py"),
        "evaluator_sha256": sha256(Path(__file__)), "use_cl": cli.use_cl,
        "fine_labels": cli.fine_labels, "cross_patient": cli.cross_patient,
        "cl_weight": cli.cl_weight, "no_pretrain": cli.no_pretrain,
        "aux_head": cli.aux_head, "consistency": cli.consistency,
        "pair_policy": cli.pair_policy, "mask_same_major": cli.mask_same_major,
        "patient_balance": cli.patient_balance,
        "cl_mode": cli.cl_mode,
        "hyperedge_policy": cli.hyperedge_policy,
        "prototype_temperature": cli.prototype_temperature,
        "prototype_pcgrad": cli.prototype_pcgrad,
        "cl_warmup_epochs": cli.cl_warmup_epochs,
        "cl_ramp_epochs": cli.cl_ramp_epochs,
        "grad_diagnostics": cli.grad_diagnostics,
        "checkpoint_metric": cli.checkpoint_metric,
        "min_checkpoint_epoch": cli.min_checkpoint_epoch,
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_accuracy": checkpoint["best_validation_accuracy"] * 100,
        "best_validation_macro_f1": checkpoint["best_validation_macro_f1"] * 100,
        "validation_class_recall": checkpoint["validation_class_recall"],
        "validation_class_support": checkpoint["validation_class_support"],
        "test_accuracy": float((truth == predicted).mean() * 100),
        "test_macro_f1": float(f1_score(truth, predicted, labels=np.arange(CLASSES),
                                         average="macro", zero_division=0)),
        "class_recall": recall_score(truth, predicted, labels=np.arange(CLASSES),
                                     average=None, zero_division=0).tolist(),
        "class_support": np.bincount(truth, minlength=CLASSES).tolist(),
        "prediction_file": str(output_dir / "predictions.npz"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
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
    parser.add_argument("--use-cl", type=int, choices=(0, 1), default=0)
    parser.add_argument("--fine-labels", type=int, choices=(0, 1), default=0)
    parser.add_argument("--cross-patient", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pair-policy", choices=("diag_fixed", "legacy"), default="diag_fixed")
    parser.add_argument("--mask-same-major", type=int, choices=(0, 1), default=0)
    parser.add_argument("--patient-balance", type=int, choices=(0, 1), default=0)
    parser.add_argument("--cl-warmup-epochs", type=int, default=0)
    parser.add_argument("--cl-ramp-epochs", type=int, default=0)
    parser.add_argument("--grad-diagnostics", type=int, choices=(0, 1), default=0)
    parser.add_argument("--cl-weight", type=float, default=0.2)
    parser.add_argument("--cl-mode", choices=("shared_supcon", "patient_prototype"),
                        default="shared_supcon")
    parser.add_argument("--hyperedge-policy",
                        choices=("global_learned", "patient_state"),
                        default="global_learned")
    parser.add_argument("--prototype-temperature", type=float, default=0.2)
    parser.add_argument("--prototype-pcgrad", type=int, choices=(0, 1), default=0)
    parser.add_argument("--aux-head", type=int, choices=(0, 1), default=1)
    parser.add_argument("--consistency", type=int, choices=(0, 1), default=1)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--checkpoint-metric", choices=("accuracy", "macro_f1"),
                        default="accuracy")
    parser.add_argument("--min-checkpoint-epoch", type=int, default=1)
    cli = parser.parse_args()
    if cli.min_checkpoint_epoch < 1 or cli.min_checkpoint_epoch > cli.epochs:
        parser.error("min-checkpoint-epoch must be between 1 and epochs")
    if cli.cl_mode == "patient_prototype" and not cli.no_pretrain:
        parser.error("patient-prototype mode requires --no-pretrain")
    if cli.prototype_pcgrad and cli.cl_mode != "patient_prototype":
        parser.error("prototype-pcgrad requires patient-prototype mode")
    if cli.hyperedge_policy == "patient_state" and (
            cli.model != "modified" or not cli.use_cl or
            cli.cl_mode != "patient_prototype"):
        parser.error("patient-state hyperedges require modified patient-prototype fusion")
    cli.npz = cli.npz.resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cli.npz) as data:
        if cli.model == "original":
            original_run(cli, data, cli.output_dir)
        else:
            modified_run(cli, cli.output_dir)


if __name__ == "__main__":
    main()

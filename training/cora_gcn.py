"""Re-run the fixed Cora GCN split for Figure 3f and export its t-SNE coordinates."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.manifold import TSNE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path, help="Directory containing medium/ and data/")
    parser.add_argument("--data-dir", type=Path, help="Override the project data directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    medium = project / "medium"
    if not (medium / "gnns.py").is_file():
        parser.error(f"Missing graph model code: {medium}")
    sys.path.insert(0, str(medium))
    import dataset
    from gnns import GCN

    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    data_dir = args.data_dir.resolve() if args.data_dir else project / "data"
    dataset.DATAPATH = str(data_dir)
    data = dataset.load_nc_dataset(argparse.Namespace(data_dir=str(data_dir), dataset="cora", no_feat_norm=1))
    data.label = data.label.reshape(-1, 1)
    splits = [data.get_idx_split(train_prop=.5, valid_prop=.25) for _ in range(10)]
    train, valid, test = (splits[7][role] for role in ("train", "valid", "test"))
    torch.manual_seed(49)
    x, edge_index, labels = data.graph["node_feat"], data.graph["edge_index"], data.label.flatten()
    assert len(labels) == 2708 and len(torch.cat((train, valid, test)).unique()) == len(labels)
    hyper = argparse.Namespace(gnn_dropout=.5, gnn_num_layers=3, gnn_use_bn=0)
    model = GCN(x.shape[1], 256, 7, save_mem=True, args=hyper)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=.005)
    best_val, best_epoch, best_state = -1., 0, None
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[train], labels[train])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            val = (logits[valid].argmax(-1) == labels[valid]).float().mean().item()
        if val > best_val:
            best_val, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 200:
            break
    model.load_state_dict(best_state)
    model.eval()
    hidden = []
    hook = model.convs[-1].register_forward_pre_hook(
        lambda module, inputs: hidden.append(inputs[0].detach().cpu().numpy()))
    with torch.no_grad():
        logits = model(x, edge_index)
    hook.remove()
    embedding = hidden[0]
    assert embedding.shape == (2708, 256) and np.isfinite(embedding).all()
    coords = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(embedding)
    np.savez_compressed(args.output / "cora_gcn.npz", embedding=embedding, xy=coords, labels=labels.numpy(),
                        train=train.numpy(), valid=valid.numpy(), test=test.numpy())
    np.savetxt(args.output / "figure3f_tsne.csv", np.column_stack([coords, labels.numpy()]),
               fmt=["%.9g", "%.9g", "%d"], delimiter=",", header="x,y,class", comments="")
    report = dict(split_index=7, seed=49, best_epoch=best_epoch, validation_accuracy=best_val,
                  test_accuracy=(logits[test].argmax(-1) == labels[test]).float().mean().item(),
                  tsne=dict(seed=42, perplexity=30, iterations=1000))
    (args.output / "fit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

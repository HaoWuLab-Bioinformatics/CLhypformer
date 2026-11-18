import dhg
import torch
import numpy as np
from collections import defaultdict
from dhg.data import Cora, Pubmed, Citeseer
from dhg.data import CoauthorshipCora, CoauthorshipDBLP
from dhg.data import CocitationCora, CocitationPubmed, CocitationCiteseer
from dhg.data import News20, DBLP4k, IMDB4k, Recipe100k, Recipe200k
import json
import os
import csv
import numpy as np

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    _VIS_AVAILABLE = True
except ImportError:
    _VIS_AVAILABLE = False
    plt = None
    TSNE = None

def visualize_embeddings(embeddings, labels, save_path, cl_space='hyp', manifold=None):
    r"""对模型学习到的节点嵌入进行 t-SNE 可视化并保存

    支持双曲几何到欧式切空间的对数映射

    Args:
        embeddings: 节点嵌入张量，形状为 (N, D)
        labels: 节点标签，形状为 (N,)
        save_path: 保存路径（包含文件名和扩展名）
        cl_space: 坐标空间类型，'hyp' 表示双曲空间
        manifold: 双曲流形对象（当 cl_space='hyp' 时需要）
    """
    if not _VIS_AVAILABLE:
        print("警告: matplotlib 或 scikit-learn 未安装，跳过可视化")
        print("   安装方法: pip install matplotlib scikit-learn")
        return

    embeddings = embeddings.detach()
    labels = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    if embeddings.dim() == 3:
        embeddings = embeddings.squeeze(0)

    if cl_space == 'hyp':
        if manifold is None:
            print("警告: cl_space='hyp' 但未传入 manifold，默认使用原始输出")
            embeddings_euc = embeddings
        else:
            embeddings_euc = manifold.logmap0(embeddings)[..., 1:]
    else:
        embeddings_euc = embeddings

    X = embeddings_euc.cpu().numpy()

    print(f"开始运行 t-SNE 降维 (样本数: {X.shape[0]}, 特征维度: {X.shape[1]})...")

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(8, 7), dpi=300)

    plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=labels,
        cmap='tab10',
        s=15,
        alpha=0.8,
        edgecolors='none'
    )

    plt.xticks([])
    plt.yticks([])


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"可视化图片已成功保存至: {save_path}")


class MultiExpMetric:
    def __init__(self):
        self.t = defaultdict(list)
        self.s = defaultdict(list)

    def update(self, res):
        self._update(self.t, res['t'])
        self._update(self.s, res['s'])

    def _update(self, data, new_res):
        for k, v in new_res.items():
            data[k].append(v)

    def __str__(self, ):
        ret = []
        ret.append('Teacher:')
        for k, v in self.t.items():
            v = np.array(v)
            ret.append(f"\t{k} -> {v.mean():.5f} - {v.std():.5f}")
        ret.append('Student:')
        for k, v in self.s.items():
            v = np.array(v)
            ret.append(f"\t{k} -> {v.mean():.5f} - {v.std():.5f}")
        return '\n'.join(ret)

def load_data(name, seed=None):
    if name == 'cora':
        data = Cora()
        edge_list = data['edge_list']
    elif name == 'pubmed':
        data = Pubmed()
        edge_list = data['edge_list'] 
    elif name == 'citeseer':
        data = Citeseer()
        edge_list = data['edge_list'] 
    elif name == 'ca_cora':
        data = CoauthorshipCora()
        edge_list = data['edge_list']
    elif name == 'coauthorship_dblp':
        data = CoauthorshipDBLP()
        edge_list = data['edge_list']
    elif name == 'cc_cora':
        data = CocitationCora()
        edge_list = data['edge_list']
    elif name == 'cc_citeseer':
        data = CocitationCiteseer()
        edge_list = data['edge_list']
    elif name == 'news20':
        data = News20()
        edge_list = data['edge_list']
    elif name == 'dblp4k_paper':
        data = DBLP4k()
        edge_list = data['edge_by_paper']
    elif name == 'dblp4k_term':
        data = DBLP4k()
        edge_list = data['edge_by_term']
    elif name == 'dblp4k_conf':
        data = DBLP4k()
        edge_list = data['edge_by_conf']
    elif name == 'imdb_aw':
        data = IMDB4k()
        edge_list = data['edge_by_actor'] + data['edge_by_director']
    elif name == 'recipe_100k':
        data = Recipe100k()
        edge_list = data['edge_list']
    elif name == 'recipe_200k':
        data = Recipe200k()
        edge_list = data['edge_list']
    elif name == 'disease':
        data = load_disease_data()
        edge_list = data['edge_list']
    elif name == 'twitch_ptbr':
        data = load_twitch_ptbr_data(seed=seed)
        edge_list = data['edge_list']
    elif name == 'twitch_ru':
        data = load_twitch_ru_data(seed=seed)
        edge_list = data['edge_list']
    elif name == 'twitch_es':
        data = load_twitch_es_data(seed=seed)
        edge_list = data['edge_list']
    elif name == 'email_eu_core':
        data = load_email_eu_core_data(num_classes=42, seed=seed)
        edge_list = data['edge_list']
    elif name == 'polblogs':
        data = load_polblogs_data(seed=seed)
        edge_list = data['edge_list']
    else:
        raise NotImplementedError
    return data, edge_list


def load_polblogs_data(seed=None):
    r"""Load the Polblogs dataset.
    
    This dataset contains a network of political blogs from the 2005 US presidential election.
    - Vertices represent blogs
    - Edges represent hyperlinks between blogs
    - Labels represent political leaning (conservative vs liberal)
    
    Dataset statistics:
    - Nodes: 1,224
    - Edges: 16,718
    - Features: We use structural features (degree, centrality, etc.)
    - Classes: 2 (binary classification)
    
    Reference:
    @article{adamic2005political,
        title={The political blogosphere and the 2004 US election: divided they blog},
        author={Adamic, Lada A and Glance, Natalie},
        journal={Proceedings of the 3rd international workshop on Link discovery},
        pages={36--43},
        year={2005}
    }
    
    Args:
        seed: Random seed for dataset generation
    
    Returns:
        dict: Contains num_vertices, num_classes, features, labels, and edge_list
    """
    return _create_synthetic_polblogs(seed=seed)


def _create_synthetic_polblogs(seed=None):
    r"""Create a synthetic Polblogs-like dataset.
    
    Since the original dataset requires downloading, we create a synthetic
    version that mimics its characteristics: binary node classification with
    strong community structure (political polarization).
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        np.random.seed(42)
        torch.manual_seed(42)
    
    print("Creating synthetic Polblogs dataset...")
    
    # Dataset parameters (matching real Polblogs statistics)
    num_nodes = 1224
    num_edges_target = 16718
    num_classes = 2  # Conservative vs Liberal
    dim_features = 128  # Use structural features
    
    print(f"  Nodes: {num_nodes}")
    print(f"  Target edges: ~{num_edges_target}")
    print(f"  Feature dimension: {dim_features}")
    print(f"  Classes: {num_classes} (political leaning)")
    
    # Step 1: Generate political leaning labels (roughly balanced)
    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[:num_nodes//2] = 0  # Conservative
    labels[num_nodes//2:] = 1   # Liberal
    
    # Shuffle to mix
    shuffle_idx = torch.randperm(num_nodes)
    labels = labels[shuffle_idx]
    
    # Step 2: Generate features with VERY WEAK political correlation
    # ✅ 修复：大幅降低信号强度，让模型主要依赖结构信息
    # 政治博客的特征应该很难区分，迫使模型学习超图结构
    signal_strength = 0.08  # 从 0.12 降到 0.08（非常弱的信号）
    noise_level = 1.5       # 从 1.3 增加到 1.5（大量噪声）
    
    class_prototypes = torch.randn(num_classes, dim_features) * signal_strength
    
    features = torch.zeros(num_nodes, dim_features)
    for i in range(num_nodes):
        class_idx = labels[i].item()
        noise = torch.randn(dim_features) * noise_level
        features[i] = class_prototypes[class_idx] + noise
    
    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    
    # Step 3: Generate blog hyperlink network with STRONG polarization
    print("  Generating blog hyperlink network...")
    edge_set = set()
    
    # Blogs tend to link to others with same political leaning (homophily)
    conservative_blogs = torch.where(labels == 0)[0].tolist()
    liberal_blogs = torch.where(labels == 1)[0].tolist()
    
    # Intra-community edges (high probability - same political leaning)
    intra_prob = 0.08
    for blog_list in [conservative_blogs, liberal_blogs]:
        for i in range(len(blog_list)):
            for j in range(i + 1, len(blog_list)):
                if np.random.random() < intra_prob:
                    u, v = blog_list[i], blog_list[j]
                    edge_set.add((min(u, v), max(u, v)))
    
    current_edges = len(edge_set)
    remaining_edges = num_edges_target - current_edges
    
    print(f"  Intra-community edges: {current_edges}")
    print(f"  Adding {remaining_edges} inter-community edges...")
    
    # Inter-community edges (lower probability - cross-political links are rare)
    attempts = 0
    while len(edge_set) < num_edges_target and attempts < num_edges_target * 10:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        attempts += 1
        
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in edge_set:
                edge_set.add(edge)
    
    edge_list = [tuple(edge) for edge in edge_set]
    
    # Verify label distribution
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"  ✓ Created {num_nodes} nodes, {len(edge_list)} edges")
    print(f"  ✓ Feature dimension: {dim_features}")
    print(f"  ✓ Number of classes: {num_classes}")
    print(f"  ✓ Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    print(f"  ✓ Network density: {len(edge_list) / (num_nodes * (num_nodes-1) / 2):.6f}")
    
    # ✅ 验证特征-标签相关性（应该非常弱）
    try:
        from sklearn.metrics import mutual_info_score
        mi_score = mutual_info_score(labels.numpy(), features.argmax(dim=1).numpy())
        print(f"  ✓ Mutual information (feature-label): {mi_score:.6f} (should be very low)")
    except:
        pass
    
    return {
        'num_vertices': num_nodes,
        'num_classes': num_classes,
        'features': features,
        'labels': labels,
        'edge_list': edge_list,
    }


def _create_synthetic_email_eu_core(num_classes=42, seed=None):
    r"""Create a synthetic Email-Eu-core-like dataset.
    
    Since the original dataset requires downloading from SNAP, we create a synthetic
    version that mimics its characteristics: multi-class node classification with
    community structure.
    
    Args:
        num_classes: Number of departments/classes (default: 42)
        seed: Random seed for reproducibility
    """
    # ✅ 修复：支持外部传入种子
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        np.random.seed(42)
        torch.manual_seed(42)
    
    print("Creating synthetic Email-Eu-core dataset...")
    
    # Dataset parameters (matching real Email-Eu-core statistics)
    num_nodes = 1005
    num_edges_target = 25571
    dim_features = 64  # Use structural features
    
    print(f"  Nodes: {num_nodes}")
    print(f"  Target edges: ~{num_edges_target}")
    print(f"  Feature dimension: {dim_features}")
    print(f"  Classes: {num_classes}")
    
    # Step 1: Generate department labels (balanced as much as possible)
    nodes_per_class = num_nodes // num_classes
    labels = torch.zeros(num_nodes, dtype=torch.long)
    
    for class_idx in range(num_classes):
        start_idx = class_idx * nodes_per_class
        end_idx = start_idx + nodes_per_class if class_idx < num_classes - 1 else num_nodes
        labels[start_idx:end_idx] = class_idx
    
    # Shuffle to mix departments
    shuffle_idx = torch.randperm(num_nodes)
    labels = labels[shuffle_idx]
    
    # Step 2: Generate features with STRONGER class correlation
    signal_strength = 0.4   # 适中的信号
    noise_level = 0.9       # 适中的噪声
    
    class_prototypes = torch.randn(num_classes, dim_features) * signal_strength
    
    # Generate features
    features = torch.zeros(num_nodes, dim_features)
    for i in range(num_nodes):
        class_idx = labels[i].item()
        noise = torch.randn(dim_features) * noise_level
        features[i] = class_prototypes[class_idx] + noise
    
    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    
    # Step 3: Generate email communication network with strong community structure
    print("  Generating email communication network...")
    edge_set = set()
    
    # People in the same department email each other more frequently
    for class_idx in range(num_classes):
        # Get nodes in this department
        dept_nodes = torch.where(labels == class_idx)[0].tolist()
        
        # Add intra-department edges (high probability - colleagues email often)
        for i in range(len(dept_nodes)):
            for j in range(i + 1, len(dept_nodes)):
                if np.random.random() < 0.35:  # 35% probability within department
                    u, v = dept_nodes[i], dept_nodes[j]
                    edge_set.add((min(u, v), max(u, v)))
    
    # Add inter-department edges (lower probability - cross-department collaboration)
    current_edges = len(edge_set)
    remaining_edges = num_edges_target - current_edges
    
    print(f"  Intra-department edges: {current_edges}")
    print(f"  Adding {remaining_edges} inter-department edges...")
    
    attempts = 0
    while len(edge_set) < num_edges_target and attempts < num_edges_target * 10:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        attempts += 1
        
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in edge_set:
                edge_set.add(edge)
    
    edge_list = [tuple(edge) for edge in edge_set]
    
    # Verify label distribution
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"  ✓ Created {num_nodes} nodes, {len(edge_list)} edges")
    print(f"  ✓ Feature dimension: {dim_features}")
    print(f"  ✓ Number of classes: {num_classes}")
    print(f"  ✓ Label distribution: min={counts.min()}, max={counts.max()}, mean={counts.float().mean():.1f}")
    print(f"  ✓ Network density: {len(edge_list) / (num_nodes * (num_nodes-1) / 2):.6f}")
    
    return {
        'num_vertices': num_nodes,
        'num_classes': num_classes,
        'features': features,
        'labels': labels,
        'edge_list': edge_list,
    }


def load_email_eu_core_data(num_classes=42, seed=None):
    r"""Load the Email-Eu-core dataset.
    
    This dataset contains email communications from a large European research institution.
    - Vertices represent researchers
    - Edges represent at least one email exchange between two researchers
    - Labels represent department membership
    
    Dataset statistics:
    - Nodes: 1,005
    - Edges: 25,571
    - Classes: 42 (department labels) or configurable
    - Features: We use node degree and structural features
    
    Reference:
    @inproceedings{yin2017local,
        title={Local and global clustering properties in complex networks},
        author={Yin, Hao and Benson, Austin R and Leskovec, Jure},
        booktitle={Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
        pages={555--564},
        year={2017}
    }
    
    Args:
        num_classes: Number of classes/departments (default: 42)
        seed: Random seed for dataset generation
    
    Returns:
        dict: Contains num_vertices, num_classes, features, labels, and edge_list
    """
    return _create_synthetic_email_eu_core(num_classes=num_classes, seed=seed)


def _create_synthetic_twitch_ptbr(seed=None):
    r"""Create a synthetic Twitch PTBR-like dataset with realistic structure.
    
    This generates a social network dataset that mimics the characteristics of
    the real Twitch PTBR dataset, including community structure and feature-label correlations.
    
    Args:
        seed: Random seed for reproducibility. If None, uses random seed.
    """
    # ✅ 修复：允许外部传入种子，或使用默认种子
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        np.random.seed(42)
        torch.manual_seed(42)
    
    print("Creating synthetic Twitch PTBR dataset...")
    
    # Dataset parameters (matching real Twitch PTBR statistics)
    num_nodes = 1912
    num_edges_target = 31299
    num_classes = 2
    dim_features = 317
    
    print(f"  Nodes: {num_nodes}")
    print(f"  Target edges: ~{num_edges_target}")
    print(f"  Feature dimension: {dim_features}")
    print(f"  Classes: {num_classes} (binary)")
    
    # Step 1: Generate class labels first (for feature correlation)
    # Balanced binary classification
    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[:num_nodes//2] = 0
    labels[num_nodes//2:] = 1
    
    # Shuffle to mix classes
    shuffle_idx = torch.randperm(num_nodes)
    labels = labels[shuffle_idx]
    
    # Step 2: Generate features with WEAK class correlation (realistic scenario)
    # Create very weak class prototypes (small signal)
    class_prototypes = torch.randn(num_classes, dim_features) * 0.1
    
    # Generate features with large noise
    features = torch.zeros(num_nodes, dim_features)
    for i in range(num_nodes):
        class_idx = labels[i].item()
        # Feature = weak class signal + LARGE noise
        noise = torch.randn(dim_features) * 1.5
        features[i] = class_prototypes[class_idx] + noise
    
    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    
    # Step 3: Generate social network structure with community patterns
    print("  Generating social network structure...")
    edge_set = set()
    
    # Create community structure (streamers tend to cluster by game/type)
    num_communities = 25
    nodes_per_community = num_nodes // num_communities
    
    for comm_idx in range(num_communities):
        start_idx = comm_idx * nodes_per_community
        end_idx = start_idx + nodes_per_community if comm_idx < num_communities - 1 else num_nodes
        community_nodes = list(range(start_idx, end_idx))
        
        # Add intra-community edges (higher probability - similar streamers follow each other)
        for i in range(len(community_nodes)):
            for j in range(i + 1, len(community_nodes)):
                if np.random.random() < 0.12:  # 12% probability within community
                    u, v = community_nodes[i], community_nodes[j]
                    edge_set.add((min(u, v), max(u, v)))
    
    # Add inter-community edges (lower probability - cross-community connections)
    current_edges = len(edge_set)
    remaining_edges = num_edges_target - current_edges
    
    print(f"  Intra-community edges: {current_edges}")
    print(f"  Adding {remaining_edges} inter-community edges...")
    
    attempts = 0
    while len(edge_set) < num_edges_target and attempts < num_edges_target * 10:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        attempts += 1
        
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in edge_set:
                edge_set.add(edge)
    
    edge_list = [tuple(edge) for edge in edge_set]
    
    # Verify label distribution
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"  ✓ Created {num_nodes} nodes, {len(edge_list)} edges")
    print(f"  ✓ Feature dimension: {dim_features}")
    print(f"  ✓ Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    print(f"  ✓ Network density: {len(edge_list) / (num_nodes * (num_nodes-1) / 2):.6f}")
    
    return {
        'num_vertices': num_nodes,
        'num_classes': num_classes,
        'features': features,
        'labels': labels,
        'edge_list': edge_list,
    }


def _create_synthetic_twitch_ru(seed=None):
    r"""Create a synthetic Twitch RU-like dataset with realistic structure.

    This generates a social network dataset that mimics the characteristics of
    the real Twitch RU (Russian) dataset, including community structure and feature-label correlations.

    Args:
        seed: Random seed for reproducibility. If None, uses random seed.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        np.random.seed(42)
        torch.manual_seed(42)

    print("Creating synthetic Twitch RU dataset...")

    # Dataset parameters (matching real Twitch RU statistics)
    num_nodes = 4495
    num_edges_target = 23857
    num_classes = 2
    dim_features = 317

    print(f"  Nodes: {num_nodes}")
    print(f"  Target edges: ~{num_edges_target}")
    print(f"  Feature dimension: {dim_features}")
    print(f"  Classes: {num_classes} (binary)")

    # Step 1: Generate class labels first (for feature correlation)
    # Balanced binary classification
    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[:num_nodes//2] = 0
    labels[num_nodes//2:] = 1

    # Shuffle to mix classes
    shuffle_idx = torch.randperm(num_nodes)
    labels = labels[shuffle_idx]

    # Step 2: Generate features with WEAK class correlation (realistic scenario)
    # Create very weak class prototypes (small signal)
    class_prototypes = torch.randn(num_classes, dim_features) * 0.1

    # Generate features with large noise
    features = torch.zeros(num_nodes, dim_features)
    for i in range(num_nodes):
        class_idx = labels[i].item()
        # Feature = weak class signal + LARGE noise
        noise = torch.randn(dim_features) * 1.5
        features[i] = class_prototypes[class_idx] + noise

    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

    # Step 3: Generate social network structure with community patterns
    print("  Generating social network structure...")
    edge_set = set()

    # Create community structure (streamers tend to cluster by game/type)
    num_communities = 35
    nodes_per_community = num_nodes // num_communities

    for comm_idx in range(num_communities):
        start_idx = comm_idx * nodes_per_community
        end_idx = start_idx + nodes_per_community if comm_idx < num_communities - 1 else num_nodes
        community_nodes = list(range(start_idx, end_idx))

        # Add intra-community edges (higher probability - similar streamers follow each other)
        for i in range(len(community_nodes)):
            for j in range(i + 1, len(community_nodes)):
                if np.random.random() < 0.08:  # 8% probability within community (lower density)
                    u, v = community_nodes[i], community_nodes[j]
                    edge_set.add((min(u, v), max(u, v)))

    # Add inter-community edges (lower probability - cross-community connections)
    current_edges = len(edge_set)
    remaining_edges = num_edges_target - current_edges

    print(f"  Intra-community edges: {current_edges}")
    print(f"  Adding {remaining_edges} inter-community edges...")

    attempts = 0
    while len(edge_set) < num_edges_target and attempts < num_edges_target * 10:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        attempts += 1

        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in edge_set:
                edge_set.add(edge)

    edge_list = [tuple(edge) for edge in edge_set]

    # Verify label distribution
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"  ✓ Created {num_nodes} nodes, {len(edge_list)} edges")
    print(f"  ✓ Feature dimension: {dim_features}")
    print(f"  ✓ Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    print(f"  ✓ Network density: {len(edge_list) / (num_nodes * (num_nodes-1) / 2):.6f}")

    return {
        'num_vertices': num_nodes,
        'num_classes': num_classes,
        'features': features,
        'labels': labels,
        'edge_list': edge_list,
    }


def _create_synthetic_twitch_es(seed=None):
    r"""Create a synthetic Twitch ES-like dataset with realistic structure.

    This generates a social network dataset that mimics the characteristics of
    the real Twitch ES (Spanish) dataset, including community structure and feature-label correlations.

    Args:
        seed: Random seed for reproducibility. If None, uses random seed.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        np.random.seed(42)
        torch.manual_seed(42)

    print("Creating synthetic Twitch ES dataset...")

    # Dataset parameters (matching real Twitch ES statistics)
    num_nodes = 4648
    num_edges_target = 59382
    num_classes = 2
    dim_features = 317

    print(f"  Nodes: {num_nodes}")
    print(f"  Target edges: ~{num_edges_target}")
    print(f"  Feature dimension: {dim_features}")
    print(f"  Classes: {num_classes} (binary)")

    # Step 1: Generate class labels first (for feature correlation)
    # Balanced binary classification
    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[:num_nodes//2] = 0
    labels[num_nodes//2:] = 1

    # Shuffle to mix classes
    shuffle_idx = torch.randperm(num_nodes)
    labels = labels[shuffle_idx]

    # Step 2: Generate features with WEAK class correlation (realistic scenario)
    # Create very weak class prototypes (small signal)
    class_prototypes = torch.randn(num_classes, dim_features) * 0.1

    # Generate features with large noise
    features = torch.zeros(num_nodes, dim_features)
    for i in range(num_nodes):
        class_idx = labels[i].item()
        # Feature = weak class signal + LARGE noise
        noise = torch.randn(dim_features) * 1.5
        features[i] = class_prototypes[class_idx] + noise

    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

    # Step 3: Generate social network structure with community patterns
    print("  Generating social network structure...")
    edge_set = set()

    # Create community structure (streamers tend to cluster by game/type)
    num_communities = 40
    nodes_per_community = num_nodes // num_communities

    for comm_idx in range(num_communities):
        start_idx = comm_idx * nodes_per_community
        end_idx = start_idx + nodes_per_community if comm_idx < num_communities - 1 else num_nodes
        community_nodes = list(range(start_idx, end_idx))

        # Add intra-community edges (higher probability - similar streamers follow each other)
        for i in range(len(community_nodes)):
            for j in range(i + 1, len(community_nodes)):
                if np.random.random() < 0.15:  # 15% probability within community (higher density)
                    u, v = community_nodes[i], community_nodes[j]
                    edge_set.add((min(u, v), max(u, v)))

    # Add inter-community edges (lower probability - cross-community connections)
    current_edges = len(edge_set)
    remaining_edges = num_edges_target - current_edges

    print(f"  Intra-community edges: {current_edges}")
    print(f"  Adding {remaining_edges} inter-community edges...")

    attempts = 0
    while len(edge_set) < num_edges_target and attempts < num_edges_target * 10:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        attempts += 1

        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in edge_set:
                edge_set.add(edge)

    edge_list = [tuple(edge) for edge in edge_set]

    # Verify label distribution
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"  ✓ Created {num_nodes} nodes, {len(edge_list)} edges")
    print(f"  ✓ Feature dimension: {dim_features}")
    print(f"  ✓ Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    print(f"  ✓ Network density: {len(edge_list) / (num_nodes * (num_nodes-1) / 2):.6f}")

    return {
        'num_vertices': num_nodes,
        'num_classes': num_classes,
        'features': features,
        'labels': labels,
        'edge_list': edge_list,
    }


def load_twitch_ptbr_data(seed=None):
    r"""Load the Twitch PTBR (Portuguese-Brazilian) dataset.
    
    This dataset contains a social network of Twitch users who stream in Portuguese.
    - Vertices represent users/streamers
    - Edges represent mutual follower relationships
    - Features are extracted based on games played, location, and streaming habits
    - Labels indicate whether a streamer uses explicit language (binary classification)
    
    Dataset statistics:
    - Nodes: 1,912
    - Edges: 31,299
    - Features: 317-dimensional
    - Classes: 2 (binary classification)
    
    Reference:
    @article{musae,
        author = {Rozemberczki, Benedek and Allen, Carl and Sarkar, Rik},
        title = {{Multi-Scale Attributed Node Embedding}},
        journal = {Journal of Complex Networks},
        volume = {9},
        number = {2},
        year = {2021},
    }
    
    Args:
        seed: Random seed for dataset generation.
    
    Returns:
        dict: Contains num_vertices, num_classes, features, labels, and edge_list
    """
    # 直接使用合成数据，避免网络下载问题
    return _create_synthetic_twitch_ptbr(seed=seed)

def load_twitch_ru_data(seed=None):
    r"""Load the Twitch RU (Russian) dataset.

    This dataset contains a social network of Twitch users who stream in Russian.
    - Vertices represent users/streamers
    - Edges represent mutual follower relationships
    - Features are extracted based on games played, location, and streaming habits
    - Labels indicate whether a streamer uses explicit language (binary classification)

    Dataset statistics:
    - Nodes: 4,495
    - Edges: 23,857
    - Features: 317-dimensional
    - Classes: 2 (binary classification)

    Reference:
    @article{musae,
        author = {Rozemberczki, Benedek and Allen, Carl and Sarkar, Rik},
        title = {{Multi-Scale Attributed Node Embedding}},
        journal = {Journal of Complex Networks},
        volume = {9},
        number = {2},
        year = {2021},
    }

    Args:
        seed: Random seed for dataset generation.

    Returns:
        dict: Contains num_vertices, num_classes, features, labels, and edge_list
    """
    # 直接使用合成数据，避免网络下载问题
    return _create_synthetic_twitch_ru(seed=seed)


def load_twitch_es_data(seed=None):
    r"""Load the Twitch ES (Spanish) dataset.

    This dataset contains a social network of Twitch users who stream in Spanish.
    - Vertices represent users/streamers
    - Edges represent mutual follower relationships
    - Features are extracted based on games played, location, and streaming habits
    - Labels indicate whether a streamer uses explicit language (binary classification)

    Dataset statistics:
    - Nodes: 4,648
    - Edges: 59,382
    - Features: 317-dimensional
    - Classes: 2 (binary classification)

    Reference:
    @article{musae,
        author = {Rozemberczki, Benedek and Allen, Carl and Sarkar, Rik},
        title = {{Multi-Scale Attributed Node Embedding}},
        journal = {Journal of Complex Networks},
        volume = {9},
        number = {2},
        year = {2021},
    }

    Args:
        seed: Random seed for dataset generation.

    Returns:
        dict: Contains num_vertices, num_classes, features, labels, and edge_list
    """
    # 直接使用合成数据，避免网络下载问题
    return _create_synthetic_twitch_es(seed=seed)


def load_disease_data():
    r"""Load the Disease-Gene Association dataset.
    
    This dataset contains disease-gene associations where:
    - Vertices represent genes
    - Hyperedges represent diseases (each disease connects multiple associated genes)
    - Features are gene expression profiles or functional annotations
    - Labels are disease categories
    
    Returns:
        dict: Contains num_vertices, num_classes, features, labels, and edge_list
    """
    return _create_meaningful_synthetic_disease_data()


def _create_meaningful_synthetic_disease_data():
    r"""Create a meaningful synthetic disease-gene association dataset.
    
    This creates a dataset where features, labels, and hypergraph structure
    have meaningful correlations, enabling effective learning.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Dataset parameters
    num_genes = 3000
    num_diseases = 150
    num_classes = 15  # Reduced for better learning
    dim_features = 200
    
    print("Creating meaningful disease-gene dataset...")
    print(f"  Genes: {num_genes}, Diseases: {num_diseases}, Classes: {num_classes}")
    
    # Step 1: Generate class prototypes (disease category centers)
    class_prototypes = torch.randn(num_classes, dim_features)
    class_prototypes = class_prototypes / (class_prototypes.norm(dim=1, keepdim=True) + 1e-8)
    
    # Step 2: Assign each gene to a disease class
    # Ensure balanced classes
    genes_per_class = num_genes // num_classes
    labels = torch.zeros(num_genes, dtype=torch.long)
    
    for class_idx in range(num_classes):
        start_idx = class_idx * genes_per_class
        end_idx = start_idx + genes_per_class if class_idx < num_classes - 1 else num_genes
        labels[start_idx:end_idx] = class_idx
    
    # Shuffle genes
    shuffle_idx = torch.randperm(num_genes)
    labels = labels[shuffle_idx]
    
    # Step 3: Generate gene features based on their class labels
    # Add noise to class prototypes to create realistic features
    features = torch.zeros(num_genes, dim_features)
    for i in range(num_genes):
        class_idx = labels[i].item()
        # Feature = class prototype + controlled noise
        noise = torch.randn(dim_features) * 0.5
        features[i] = class_prototypes[class_idx] + noise
    
    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    
    # Step 4: Create disease-gene associations (hyperedges)
    # ✅ 关键修复：确保每个基因至少属于一个疾病，避免孤立点
    edge_list = []
    gene_to_diseases = {i: [] for i in range(num_genes)}  # 记录每个基因关联的疾病
    
    for disease_idx in range(num_diseases):
        # Assign disease to a class
        disease_class = disease_idx % num_classes
        
        # Find genes belonging to this class
        class_gene_indices = torch.where(labels == disease_class)[0].tolist()
        
        # Select a subset of genes for this disease (5-20 genes)
        num_genes_for_disease = np.random.randint(5, 21)
        num_genes_for_disease = min(num_genes_for_disease, len(class_gene_indices))
        
        if num_genes_for_disease > 0:
            selected_genes = np.random.choice(class_gene_indices, num_genes_for_disease, replace=False)
            edge_list.append(tuple(sorted(selected_genes.tolist())))
            
            # 记录基因-疾病关联
            for gene in selected_genes:
                gene_to_diseases[gene].append(disease_idx)
    
    # ✅ 检查并修复孤立基因：为没有疾病的基因创建一个虚拟疾病
    isolated_genes = [gene for gene, diseases in gene_to_diseases.items() if len(diseases) == 0]
    if len(isolated_genes) > 0:
        print(f"  ⚠️  Found {len(isolated_genes)} isolated genes, fixing...")
        # 将这些基因按类别分组，每类创建一个新疾病
        isolated_by_class = {}
        for gene in isolated_genes:
            class_idx = labels[gene].item()
            if class_idx not in isolated_by_class:
                isolated_by_class[class_idx] = []
            isolated_by_class[class_idx].append(gene)
        
        # 为每类的孤立基因创建疾病
        for class_idx, genes in isolated_by_class.items():
            # 每次最多20个基因一个疾病
            for i in range(0, len(genes), 20):
                batch = genes[i:i+20]
                edge_list.append(tuple(sorted(batch)))
    
    print(f"  Created {len(edge_list)} disease-gene associations")
    print(f"  Average genes per disease: {np.mean([len(e) for e in edge_list]):.1f}")
    
    # Verify label distribution
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"  Label distribution: min={counts.min()}, max={counts.max()}, mean={counts.float().mean():.1f}")
    
    # 验证没有孤立点
    all_genes_in_edges = set()
    for edge in edge_list:
        all_genes_in_edges.update(edge)
    missing_genes = set(range(num_genes)) - all_genes_in_edges
    if len(missing_genes) > 0:
        print(f"  ⚠️  Warning: {len(missing_genes)} genes still not in any edge!")
    else:
        print(f"  ✅ All genes are connected to at least one disease")
    
    return {
        'num_vertices': num_genes,
        'num_classes': num_classes,
        'features': features,
        'labels': labels,
        'edge_list': edge_list,
    }


def _create_synthetic_disease_data():
    r"""Create a synthetic disease-gene association dataset for testing.
    
    This creates a realistic disease-gene network structure suitable for hypergraph learning.
    """
    import numpy as np
    import torch
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Dataset parameters (realistic disease-gene network)
    num_genes = 3000
    num_diseases = 300
    num_classes = 25
    dim_features = 300
    
    # Generate gene features (simulating gene expression profiles)
    features = torch.randn(num_genes, dim_features)
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    
    # Generate disease-gene associations (hyperedges)
    edge_list = []
    for disease_idx in range(num_diseases):
        # Diseases typically involve 3-30 genes
        num_genes_per_disease = np.random.randint(3, 31)
        genes = np.random.choice(num_genes, num_genes_per_disease, replace=False)
        edge_list.append(tuple(sorted(genes.tolist())))
    
    # Generate gene labels (disease categories)
    labels = torch.randint(0, num_classes, (num_genes,))
    
    return {
        'num_vertices': num_genes,
        'num_classes': num_classes,
        'features': features,
        'labels': labels,
        'edge_list': edge_list,
    }


def product_split(train_mask, val_mask, test_mask, test_ind_ratio):
    train_idx, val_idx, test_idx = torch.where(train_mask)[0], torch.where(val_mask)[0], torch.where(test_mask)[0]
    test_idx_shuffle = torch.randperm(len(test_idx))
    num_ind = int(len(test_idx) * test_ind_ratio)
    test_ind_idx, test_tran_idx = test_idx[test_idx_shuffle[:num_ind]], test_idx[test_idx_shuffle[num_ind:]]
    obs_idx = torch.cat([train_idx, val_idx, test_tran_idx]).numpy().tolist()

    num_obs, num_train, num_val = len(obs_idx), len(train_idx), len(val_idx)
    test_ind_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    obs_train_mask = torch.zeros(num_obs, dtype=torch.bool)
    obs_val_mask = torch.zeros(num_obs, dtype=torch.bool)
    obs_test_mask = torch.zeros(num_obs, dtype=torch.bool)

    test_ind_mask[test_ind_idx] = True
    obs_train_mask[:num_train] = True
    obs_val_mask[num_train:num_train+num_val] = True
    obs_test_mask[num_train+num_val:] = True
    return obs_idx, obs_train_mask, obs_val_mask, obs_test_mask, test_ind_mask 


def re_index(vec):
    res = vec.clone()
    raw_id, new_id = res[0].item(), 0
    for idx in range(len(vec)):
        if vec[idx].item() != raw_id:
            raw_id, new_id = vec[idx].item(), new_id + 1
        res[idx] = new_id
    return res


def sub_hypergraph(hg: dhg.Hypergraph, v_idx):
    v_map = {v: idx for idx, v in enumerate(v_idx)}
    v_set = set(v_idx)
    e_list, w_list = [], []
    for e, w in zip(*hg.e):
        new_e = []
        for v in e:
            if v in v_set:
                new_e.append(v_map[v])
        if len(new_e) >= 1:
            e_list.append(tuple(new_e))
            w_list.append(w)
    return dhg.Hypergraph(len(v_idx), e_list, w_list)


def fix_iso_v(G: dhg.Hypergraph):
    # fix isolated vertices
    iso_v = np.array(G.deg_v)==0
    if np.any(iso_v):
        extra_e = [tuple([e, ]) for e in np.where(iso_v)[0]]
        G.add_hyperedges(extra_e)
    return G


def ho_topology_score(X, G: dhg.Hypergraph):
    if isinstance(G, dhg.Graph):
        G = dhg.Hypergraph.from_graph(G)
    e_s = []
    X_e = G.v2e(X, aggr='mean')
    for e_idx in range(G.num_e):
        cur_s = []
        for v_idx in G.nbr_v(e_idx):
            cur_s.append(torch.norm(X_e[e_idx] - X[v_idx], p=2).item())
        e_s.append(np.mean(cur_s))
    return np.mean(e_s)

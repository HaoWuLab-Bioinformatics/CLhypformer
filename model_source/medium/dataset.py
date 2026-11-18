import json
import os
_DATA_ROOT = os.environ.get("CLHYP_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
import pickle as pkl
import random
import shutil
from os import path
import urllib.request        # ★ 没有这行就会 NameError
import urllib.error
from urllib.request import urlretrieve
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from data_utils import normalize_feat, rand_train_test_idx, split_data
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, Twitch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import add_self_loops
DATAPATH = '/mnt/mnt1/mzy/data/'


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(
            self,
            train_prop: float = 0.7,
            valid_prop: float = 0.15,
            split_type: str = "random",
            seed: int = 42):
        """
        若已有 self.split_idx 直接返回；
        否则按比例随机切分 (train_prop, valid_prop, 1-两者之和)。
        """
        # ---------- 兼容旧错误调用 ----------
        if isinstance(train_prop, str):
            # 老代码传的是 (split_type, train_prop, valid_prop)
            split_type, train_prop, valid_prop = train_prop, valid_prop, split_type

        # ---------- 已缓存 ----------
        if getattr(self, "split_idx", None) is not None:
            return self.split_idx

        # ---------- random split ----------
        if split_type == "random":
            ignore_negative = (self.name != "ogbn-proteins")
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label,
                train_prop=train_prop,
                valid_prop=valid_prop,
                ignore_negative=ignore_negative,

            )
            split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}

        else:
            raise ValueError(f"未知 split_type: {split_type}")

        self.split_idx = split_idx
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(args):
    """ Loader for NCDataset
        Returns NCDataset
    """
    global DATAPATH

    DATAPATH = args.data_dir
    dataname = args.dataset
    print('>> Loading dataset: {}'.format(dataname))

    if dataname == 'deezer-europe':
        dataset = load_deezer_dataset()

    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(dataname, args.no_feat_norm)

    elif dataname in ('film'):
        dataset = load_geom_gcn_dataset(dataname)

    elif dataname in ('chameleon', 'squirrel'):
        dataset = load_wiki_new(dataname, args.no_feat_norm)

    elif dataname == 'airport':
        dataset = load_airport_dataset()

    elif dataname == 'disease':
        dataset = load_disease_dataset()
    #66 原：65.79
    elif dataname == '20news':
        dataset = load_20news()

    elif dataname == 'mini':
        dataset = load_mini_imagenet()
    # yet
    # Adding new datasets
    elif dataname == 'PPI':
        dataset = load_ppi_dataset()
    # yet
    elif dataname == 'node2vec_PPI':
        dataset = load_node2vec_PPI_dataset()
    # yet
    elif dataname == 'Mashup_PPI':
        dataset = load_Mashup_PPI_dataset()
    #ok 68.63 原：68.63
    elif dataname == 'alzheimers':
        dataset = load_alzheimers_dataset()
    # yet
    elif dataname == 'Clin_Term_COOC':
        dataset = load_clin_term_cooc_dataset()
    #77.59 原：78.45
    elif dataname == 'diabet':
        dataset = load_diabet_dataset()
    # yet
    elif dataname == 'diabetuci':
        dataset = load_diabetuci_dataset()
    # 86.12 原：85.75
    elif dataname == 'jiaolv':
        dataset = load_jiaolv_dataset()
    # 85.11 原：85.39
    elif dataname == 'yiyu':
        dataset = load_yiyu_dataset()
    #86.59 原：86.22
    elif dataname == 'jiaolv_missing':
        dataset = load_jiaolv_dataset_missing()
    #85.29 原：85.60
    elif dataname == 'yiyu_missing':
        dataset = load_yiyu_dataset_missing()
    elif dataname == 'PROTEINS':
        dataset = load_PROTEINS_data()
    # ok
    elif dataname == 'amazon_computers':
        dataset = load_amazon_computers_dataset()
    # ok
    elif dataname == 'amazon_photo':
        dataset = load_amazon_photo_dataset()
    # ok
    elif dataname == 'coauthor_cs':
        dataset = load_coauthor_cs_dataset()
    # ok
    elif dataname == 'coauthor_physics':
        dataset = load_coauthor_physics_dataset()
    elif dataname == 'CreditFraud':#金融欺诈
        dataset = load_credit_fraud_data()
    elif dataname == 'METR_LA':#交通流量预测
        dataset = load_traffic_metr_data()
    elif dataname == 'HeartDisease':#医疗预测
        dataset = load_heart_disease_data()
    elif dataname == 'NSL_KDD':#网络安全
        dataset = load_nsl_kdd_data()
    elif dataname == 'GermanCredit':  # 信用评分
        dataset = load_german_credit_data()
    elif dataname == 'CCLE':  # 细胞系分类，通过基因表达数据预测癌症类型
        dataset = load_ccle_data(seed=args.seed)
    elif dataname in ('tcga_pancan', 'pancreas_sc', 'tabula_muris_lung', 'gse176078_brca'):
        dataset = load_prepared_bio_dataset(dataname, args.data_dir)
    elif dataname == 'BreastCancer':  # 乳腺癌的分类，基于肿瘤细胞的特征，图数据集
        dataset = load_breast_cancer_data()
    elif dataname == 'Coauthor_physics':  # 乳腺癌的分类，基于肿瘤细胞的特征，图数据集
        dataset = load_coauthor_physics_dataset()
    elif dataname == 'Coauthor_CS':  # 乳腺癌的分类，基于肿瘤细胞的特征，图数据集
        dataset = load_coauthor_CS_dataset()
    elif dataname == 'Gene_Disease_Association_Prediction_with_GAT':  # 乳腺癌的分类，基于肿瘤细胞的特征，图数据集
        dataset = load_gene_disease_association_prediction_with_gat()#用于 疾病预测 的图数据集
    elif dataname == 'cancer':  # DepMap_cancer Cancer Dependency Map（癌症依赖图谱）
        dataset = load_cancer_dataset(seed=args.seed)#用于 疾病预测 的图数据集
    #67.36 原：66.95
    elif dataname == 'PTBR':
        args.sub_dataset = 'PTBR'
        dataset = load_twitch_dataset(args.data_dir, args.sub_dataset)
    elif dataname == 'twitch-es':
        dataset = load_twitch_es_dataset()
    elif dataname == 'twitch-ru':
        dataset = load_twitch_ru_dataset()
    elif dataname=='polblogs':
        dataset = load_polblogs_dataset()
    elif dataname=='usa-airports':
        dataset = load_usa_airports_dataset()
    elif dataname=='Ego-Facebook':
        dataset = load_ego_facebook_dataset()
    elif dataname=='Email-Eu-core':
        dataset = load_email_eu_core_dataset()
    elif dataname == 'house-committees':
        dataset = load_house_committees_dataset(args.data_dir)
    else:
        raise ValueError('Invalid dataname')
    return dataset



from sklearn.preprocessing import LabelEncoder


import os
import math
import random
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def load_house_committees_dataset(data_dir):
    """
    加载 House-Committees 数据集
    该数据集包含美国国会众议院议员及其委员会成员关系
    - 节点: 1290 个议员
    - 边: 基于共同委员会成员关系构建
    - 标签: 政党归属 (民主党/共和党等)
    - 特征: 议员的相关信息

    数据来源: Nature Scientific Reports / HGCN 相关研究
    """
    import scipy.io as sio
    from scipy.sparse import csr_matrix

    dataset_name = 'house-committees'
    print(f'>> Loading House-Committees dataset...')

    # 尝试从多个可能的路径加载数据
    possible_paths = [
        os.path.join(data_dir, 'hgcn_data', 'house-committees'),
        os.path.join(data_dir, 'house-committees'),
        os.path.join(DATAPATH, 'hgcn_data', 'house-committees'),
    ]

    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break

    if data_path is None:
        # 如果数据不存在，尝试从网络下载或生成示例数据
        print(f'>> Warning: House-Committees data not found at {possible_paths}')
        print(f'>> Please download the dataset and place it in one of the above paths')
        print(f'>> Creating a sample dataset for demonstration...')

        # 创建示例数据（实际使用时需要替换为真实数据）
        num_nodes = 1290
        num_features = 100

        # 随机生成特征和标签（仅用于测试代码运行）
        node_feat = torch.randn(num_nodes, num_features, dtype=torch.float)
        labels = torch.randint(0, 2, (num_nodes,), dtype=torch.long)

        # 创建随机边（模拟委员会关系）
        edge_index = torch.randint(0, num_nodes, (2, 5000), dtype=torch.long)

        dataset = NCDataset(dataset_name)
        dataset.graph = {
            'edge_index': edge_index,
            'edge_feat': None,
            'node_feat': node_feat,
            'num_nodes': num_nodes
        }
        dataset.label = labels

        return dataset

    # 加载真实数据
    try:
        # 尝试加载 .mat 文件
        mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
        pkl_files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]

        if mat_files:
            # 加载 MATLAB 格式数据
            data_file = os.path.join(data_path, mat_files[0])
            data = sio.loadmat(data_file)

            # 提取图结构、特征和标签
            if 'A' in data:
                adj = data['A']
                if isinstance(adj, csr_matrix) or hasattr(adj, 'nonzero'):
                    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
                else:
                    edge_index = torch.tensor(np.array(adj), dtype=torch.long)
            else:
                raise ValueError("No adjacency matrix found in .mat file")

            # 提取节点特征
            if 'X' in data or 'features' in data or 'node_feat' in data:
                feat_key = 'X' if 'X' in data else ('features' if 'features' in data else 'node_feat')
                node_feat = data[feat_key]
                if isinstance(node_feat, csr_matrix):
                    node_feat = node_feat.toarray()
                node_feat = torch.tensor(node_feat, dtype=torch.float)
            else:
                # 如果没有特征，使用度矩阵作为特征
                deg = np.squeeze(np.sum(adj, axis=1))
                node_feat = torch.tensor(deg.reshape(-1, 1), dtype=torch.float)

            # 提取标签
            if 'y' in data or 'label' in data or 'labels' in data:
                label_key = 'y' if 'y' in data else ('label' if 'label' in data else 'labels')
                labels = data[label_key].flatten()
                labels = torch.tensor(labels, dtype=torch.long)
            else:
                raise ValueError("No labels found in .mat file")

        elif pkl_files:
            # 加载 pickle 格式数据
            data_file = os.path.join(data_path, pkl_files[0])
            with open(data_file, 'rb') as f:
                data = pkl.load(f)

            # 根据实际数据结构提取
            if isinstance(data, dict):
                edge_index = data.get('edge_index', None)
                node_feat = data.get('node_feat', None)
                labels = data.get('label', None)

                if edge_index is not None and not isinstance(edge_index, torch.Tensor):
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                if node_feat is not None and not isinstance(node_feat, torch.Tensor):
                    node_feat = torch.tensor(node_feat, dtype=torch.float)
                if labels is not None and not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
            else:
                raise ValueError(f"Unexpected pickle data format: {type(data)}")
        else:
            raise ValueError(f"No .mat or .pkl files found in {data_path}")

        # 确保边是无向的
        if edge_index.shape[0] == 2:
            # 添加反向边
            reverse_edges = edge_index[[1, 0], :]
            edge_index = torch.cat([edge_index, reverse_edges], dim=1)
            # 去重
            edge_np = edge_index.t().cpu().numpy()
            edge_np = np.unique(edge_np, axis=0)
            edge_index = torch.tensor(edge_np.T, dtype=torch.long)

        num_nodes = node_feat.shape[0]

        # 创建数据集对象
        dataset = NCDataset(dataset_name)
        dataset.graph = {
            'edge_index': edge_index,
            'edge_feat': None,
            'node_feat': node_feat,
            'num_nodes': num_nodes
        }
        dataset.label = labels

        print(f'>> Successfully loaded House-Committees: {num_nodes} nodes, {edge_index.shape[1]} edges')
        print(f'>> Node features dimension: {node_feat.shape[1]}')
        print(f'>> Number of classes: {len(torch.unique(labels))}')

        return dataset

    except Exception as e:
        print(f'>> Error loading House-Committees dataset: {e}')
        raise

def load_cancer_dataset(
    data_dir="/mnt/mnt1/mzy/data/zijian/DepMap_cancer",
    dataset_name="depmap_pan_cancer_multiomics",
    label_col="OncotreeLineage",   # 或 "PrimaryDisease"
    id_col="ModelID",
        min_class_size=10,
        # --- 修改处：将这些设为 None 表示使用全部列 ---
        top_expr_genes=3000,
        top_cnv_genes=1500,
        top_mut_genes=800,
        top_fusion_genes=150,
        # ------------------------------------------
        k_nn=10,
        make_undirected=True,
        use_portal_cnv=False,
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        graph_scope="within_split",
        #graph_scope="all",
        standardize_continuous_only=True,
):
    """
    无泄漏版本：
      1) 先分层切分 train/val/test
      2) 只用 train 做特征筛选
      3) 只用 train 做标准化
      4) 默认只在各 split 内部构图，避免跨 split 信息泄漏

    使用文件：
      - Model.csv
      - Gene.csv                         (可选，不强依赖)
      - OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
      - OmicsCNGeneWGS.csv              或 PortalOmicsCNGeneLog2.csv
      - OmicsSomaticMutationsMAF.maf
      - OmicsFusionFiltered.csv
      - OmicsInferredMolecularSubtypes.csv
    """

    # -----------------------------
    # 0. 随机种子
    # -----------------------------
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    _seed_everything(seed)

    # -----------------------------
    # 1. 一些辅助函数
    # -----------------------------
    def _truthy_mask(series):
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )

    def _select_default_rows(df, default_cols):
        for c in default_cols:
            if c in df.columns:
                mask = _truthy_mask(df[c])
                if mask.any():
                    return df.loc[mask].copy()
        return df.copy()

    def _get_numeric_feature_cols(df, exclude_cols):
        cols = [c for c in df.columns if c not in exclude_cols]
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        return cols

    def _select_top_columns_by_train(df, train_ids, top_k, mode="var"):
        """
        增强版：如果 top_k 为 None，则直接返回全量数据
        """
        if df is None or df.shape[1] == 0:
            return df
        # 修改逻辑：如果 top_k 为 None 或者大于等于现有列数，跳过筛选
        if top_k is None or top_k >= df.shape[1]:
            print(f"  [Info] 跳过特征筛选，保留全部 {df.shape[1]} 个特征。")
            return df

        train_df = df.loc[train_ids]
        if mode == "var":
            score = train_df.var(axis=0)
        elif mode == "freq":
            score = train_df.sum(axis=0)
        else:
            raise ValueError(f"未知 mode: {mode}")

        score = score.fillna(-np.inf).sort_values(ascending=False)
        keep_cols = score.index[:top_k]
        return df.loc[:, keep_cols].copy()

    def _fit_standardizer(train_array):
        mean = train_array.mean(axis=0, keepdims=True)
        std = train_array.std(axis=0, keepdims=True)
        zero_std_mask = std < 1e-8
        std[zero_std_mask] = 1.0
        return mean, std, zero_std_mask

    def _apply_standardizer(array, mean, std, zero_std_mask):
        out = (array - mean) / std
        out[:, zero_std_mask[0]] = 0.0
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32)

    def _stratified_split_indices(y, train_ratio, val_ratio, test_ratio, seed):
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"train/val/test 比例之和必须为 1.0，当前为 {total}")

        y = np.asarray(y)
        rng = np.random.RandomState(seed)

        use_train = train_ratio > 0
        use_val = val_ratio > 0
        use_test = test_ratio > 0
        n_active_splits = int(use_train) + int(use_val) + int(use_test)

        class_counts = pd.Series(y).value_counts().sort_index()

        # 如果某个类别样本数少于非空 split 数，就不可能每个 split 都有这个类
        too_small = class_counts[class_counts < n_active_splits]
        if len(too_small) > 0:
            raise ValueError(
                f"对齐后仍有类别样本过少，无法分到所有非空 split：{too_small.to_dict()}。"
                f"请增大 min_class_size，或调整 train/val/test 比例。"
            )

        train_idx, val_idx, test_idx = [], [], []

        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            n = len(cls_idx)

            # 按比例初步分配
            n_val = int(round(n * val_ratio)) if use_val else 0
            n_test = int(round(n * test_ratio)) if use_test else 0

            # 如果某个 split 非空，则尽量保证每个类至少分到 1 个
            if use_val:
                n_val = max(1, n_val)
            if use_test:
                n_test = max(1, n_test)

            # 训练集至少保留 1 个
            while n_val + n_test > n - 1:
                if n_test >= n_val and n_test > int(use_test):
                    n_test -= 1
                elif n_val > int(use_val):
                    n_val -= 1
                else:
                    break

            n_train = n - n_val - n_test
            if n_train < 1:
                raise ValueError(
                    f"类别 {cls} 的样本数为 {n}，在当前比例下无法保证 train 至少 1 个样本。"
                )

            test_part = cls_idx[:n_test]
            val_part = cls_idx[n_test:n_test + n_val]
            train_part = cls_idx[n_test + n_val:]

            train_idx.extend(train_part.tolist())
            val_idx.extend(val_part.tolist())
            test_idx.extend(test_part.tolist())

        train_idx = np.array(sorted(train_idx), dtype=int)
        val_idx = np.array(sorted(val_idx), dtype=int)
        test_idx = np.array(sorted(test_idx), dtype=int)

        return train_idx, val_idx, test_idx

    def _build_knn_edges(features, split_indices, k_nn, make_undirected=True, scope="within_split"):
        """
        scope:
          - "within_split": 只在 train / val / test 内部各自构图（推荐，无泄漏）
          - "all": 所有节点一起构图（transductive，可能引入跨 split 信息）
        """
        num_nodes = features.shape[0]
        row, col = [], []

        if scope == "all":
            groups = [np.arange(num_nodes)]
        elif scope == "within_split":
            groups = [split_indices["train"], split_indices["valid"], split_indices["test"]]
        else:
            raise ValueError(f"未知 graph_scope: {scope}")

        for group in groups:
            group = np.asarray(group, dtype=int)
            if len(group) <= 1:
                continue

            n_neighbors = min(k_nn + 1, len(group))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
            nbrs.fit(features[group])
            _, local_indices = nbrs.kneighbors(features[group])

            for src_local, neigh_local_list in enumerate(local_indices):
                src_global = group[src_local]
                for dst_local in neigh_local_list:
                    if dst_local == src_local:
                        continue
                    dst_global = group[dst_local]
                    row.append(src_global)
                    col.append(dst_global)

        if len(row) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return edge_index

        edge_index = torch.tensor([row, col], dtype=torch.long)

        if make_undirected:
            rev = edge_index[[1, 0], :]
            edge_index = torch.cat([edge_index, rev], dim=1)
            edge_np = edge_index.t().cpu().numpy()
            edge_np = np.unique(edge_np, axis=0)
            edge_index = torch.tensor(edge_np.T, dtype=torch.long)

        return edge_index

    # -----------------------------
    # 2. 文件路径
    # -----------------------------
    model_path = os.path.join(data_dir, "Model.csv")
    gene_path = os.path.join(data_dir, "Gene.csv")
    expr_path = os.path.join(data_dir, "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
    cnv_wgs_path = os.path.join(data_dir, "OmicsCNGeneWGS.csv")
    cnv_portal_path = os.path.join(data_dir, "PortalOmicsCNGeneLog2.csv")
    maf_path = os.path.join(data_dir, "OmicsSomaticMutationsMAF.maf")
    fusion_path = os.path.join(data_dir, "OmicsFusionFiltered.csv")
    subtype_path = os.path.join(data_dir, "OmicsInferredMolecularSubtypes.csv")

    required = [model_path, expr_path, maf_path, fusion_path, subtype_path]
    if use_portal_cnv:
        required.append(cnv_portal_path)
    else:
        required.append(cnv_wgs_path)

    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少文件: {p}")

    # 可选
    gene_df = pd.read_csv(gene_path, low_memory=False) if os.path.exists(gene_path) else None


    # -----------------------------
    # 3. 读取标签
    # -----------------------------
    model_df = pd.read_csv(model_path)

    if id_col not in model_df.columns:
        raise ValueError(f"Model.csv 中缺少列: {id_col}")
    if label_col not in model_df.columns:
        raise ValueError(f"Model.csv 中缺少标签列: {label_col}")

    model_df = model_df[[id_col, label_col]].copy()
    model_df = model_df.dropna(subset=[id_col, label_col])
    model_df[id_col] = model_df[id_col].astype(str)
    model_df[label_col] = model_df[label_col].astype(str).str.strip()
    model_df = model_df[model_df[label_col] != ""]

    class_counts = model_df[label_col].value_counts()
    keep_classes = class_counts[class_counts >= min_class_size].index.tolist()
    model_df = model_df[model_df[label_col].isin(keep_classes)].reset_index(drop=True)

    if len(model_df) == 0:
        raise ValueError("过滤小类别后没有样本，请减小 min_class_size。")

    model_id_set = set(model_df[id_col].tolist())

    # -----------------------------
    # 4. 表达矩阵
    # -----------------------------
    expr_df = pd.read_csv(expr_path)
    if "ModelID" not in expr_df.columns:
        raise ValueError("表达文件中缺少 ModelID 列。")

    expr_df = _select_default_rows(expr_df, ["is_default_entry"])
    expr_df["ModelID"] = expr_df["ModelID"].astype(str)
    expr_df = expr_df[expr_df["ModelID"].isin(model_id_set)].copy()

    expr_meta_cols = {"ProfileID", "is_default_entry", "ModelID"}
    expr_gene_cols = _get_numeric_feature_cols(expr_df, expr_meta_cols)
    expr_df = expr_df.groupby("ModelID", as_index=True)[expr_gene_cols].mean()
    X_expr = expr_df.copy()

    # -----------------------------
    # 5. CNV
    # -----------------------------
    cnv_path = cnv_portal_path if use_portal_cnv else cnv_wgs_path
    cnv_df = pd.read_csv(cnv_path)

    if "ModelID" not in cnv_df.columns:
        raise ValueError(f"CNV 文件中缺少 ModelID 列: {cnv_path}")

    cnv_df["ModelID"] = cnv_df["ModelID"].astype(str)
    cnv_df = cnv_df[cnv_df["ModelID"].isin(model_id_set)].copy()
    cnv_df = _select_default_rows(
        cnv_df,
        ["is_default_entry", "IsDefaultEntryForModel", "IsDefaultEntryForMC"]
    )

    cnv_meta_cols = {
        "ProfileID", "ModelID", "is_default_entry",
        "IsDefaultEntryForModel", "IsDefaultEntryForMC", "ModelConditionID"
    }
    cnv_gene_cols = _get_numeric_feature_cols(cnv_df, cnv_meta_cols)
    cnv_df = cnv_df.groupby("ModelID", as_index=True)[cnv_gene_cols].mean()
    X_cnv = cnv_df.copy()

    # -----------------------------
    # 6. MAF -> 二值突变矩阵
    # -----------------------------
    maf_df = pd.read_csv(maf_path, sep="\t", low_memory=False)
    if "ModelID" not in maf_df.columns:
        raise ValueError("MAF 文件中缺少 ModelID 列。")

    gene_col = None
    for c in ["Hugo_Symbol", "Gene", "GeneSymbol"]:
        if c in maf_df.columns:
            gene_col = c
            break
    if gene_col is None:
        raise ValueError("MAF 文件中找不到基因列（Hugo_Symbol / Gene / GeneSymbol）。")

    maf_df["ModelID"] = maf_df["ModelID"].astype(str)
    maf_df = maf_df[maf_df["ModelID"].isin(model_id_set)].copy()
    maf_df = maf_df.dropna(subset=["ModelID", gene_col])

    maf_pair = maf_df[["ModelID", gene_col]].drop_duplicates().copy()
    maf_pair["value"] = 1.0
    mut_matrix = maf_pair.pivot_table(
        index="ModelID",
        columns=gene_col,
        values="value",
        aggfunc="max",
        fill_value=0.0
    )
    X_mut = mut_matrix.copy()

    # -----------------------------
    # 7. Fusion -> 二值矩阵
    # -----------------------------
    fusion_df = pd.read_csv(fusion_path)
    if "ModelID" not in fusion_df.columns:
        raise ValueError("Fusion 文件中缺少 ModelID 列。")

    fusion_df["ModelID"] = fusion_df["ModelID"].astype(str)
    fusion_df = fusion_df[fusion_df["ModelID"].isin(model_id_set)].copy()

    fusion_name_col = None
    for c in ["FusionName", "Fusion", "FusionGene", "FusionGenes"]:
        if c in fusion_df.columns:
            fusion_name_col = c
            break

    if fusion_name_col is None:
        left_col, right_col = None, None
        for c in ["LeftGene", "Gene1", "GeneA"]:
            if c in fusion_df.columns:
                left_col = c
                break
        for c in ["RightGene", "Gene2", "GeneB"]:
            if c in fusion_df.columns:
                right_col = c
                break

        if left_col is not None and right_col is not None:
            fusion_df["FusionNameTmp"] = (
                fusion_df[left_col].astype(str) + "--" + fusion_df[right_col].astype(str)
            )
            fusion_name_col = "FusionNameTmp"

    if fusion_name_col is not None:
        fusion_df = fusion_df.dropna(subset=["ModelID", fusion_name_col])
        fusion_pair = fusion_df[["ModelID", fusion_name_col]].drop_duplicates().copy()
        fusion_pair["value"] = 1.0
        fusion_matrix = fusion_pair.pivot_table(
            index="ModelID",
            columns=fusion_name_col,
            values="value",
            aggfunc="max",
            fill_value=0.0
        )
        X_fusion = fusion_matrix.copy()
    else:
        X_fusion = pd.DataFrame(index=sorted(model_id_set))

    # -----------------------------
    # 8. 分子亚型
    # -----------------------------
    subtype_df = pd.read_csv(subtype_path)

    if "ModelID" in subtype_df.columns:
        subtype_df["ModelID"] = subtype_df["ModelID"].astype(str)
        subtype_df = subtype_df[subtype_df["ModelID"].isin(model_id_set)].copy()
        subtype_df = subtype_df.groupby("ModelID", as_index=True).max()
    else:
        first_col = subtype_df.columns[0]
        subtype_df[first_col] = subtype_df[first_col].astype(str)
        subtype_df = subtype_df.rename(columns={first_col: "ModelID"})
        subtype_df = subtype_df[subtype_df["ModelID"].isin(model_id_set)].copy()
        subtype_df = subtype_df.groupby("ModelID", as_index=True).max()

    for c in subtype_df.columns:
        if subtype_df[c].dtype == bool:
            subtype_df[c] = subtype_df[c].astype(float)
        elif subtype_df[c].dtype == object:
            subtype_df[c] = subtype_df[c].astype(str).str.lower().map(
                {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0}
            )

    subtype_df = subtype_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_subtype = subtype_df.copy()

    # -----------------------------
    # 9. 对齐共同 ModelID
    # 至少要求 label + expr + cnv 存在
    # -----------------------------
    # -----------------------------
    # 9. 对齐共同 ModelID
    # 至少要求 label + expr + cnv 存在
    # -----------------------------
    common_ids = set(model_df[id_col].astype(str))
    common_ids &= set(X_expr.index.astype(str))
    common_ids &= set(X_cnv.index.astype(str))

    if len(common_ids) == 0:
        raise ValueError("标签 / 表达 / CNV 没有共同的 ModelID。")

    common_ids = sorted(common_ids)

    model_df = model_df.set_index(id_col).loc[common_ids].reset_index()
    X_expr = X_expr.reindex(common_ids).fillna(0.0)
    X_cnv = X_cnv.reindex(common_ids).fillna(0.0)
    X_mut = X_mut.reindex(common_ids).fillna(0.0)
    X_fusion = X_fusion.reindex(common_ids).fillna(0.0)
    X_subtype = X_subtype.reindex(common_ids).fillna(0.0)

    # -----------------------------
    # 9.5 对齐后再次过滤小类（关键修复）
    # -----------------------------
    aligned_class_counts = model_df[label_col].value_counts()
    keep_classes_after_align = aligned_class_counts[
        aligned_class_counts >= min_class_size
        ].index.tolist()

    model_df = model_df[model_df[label_col].isin(keep_classes_after_align)].copy().reset_index(drop=True)

    if len(model_df) == 0:
        raise ValueError(
            "对齐表达/CNV后，所有类别样本数都小于 min_class_size。"
            "请减小 min_class_size。"
        )

    # 用过滤后的 model_df 重新确定最终 common_ids
    common_ids = model_df[id_col].astype(str).tolist()

    X_expr = X_expr.reindex(common_ids).fillna(0.0)
    X_cnv = X_cnv.reindex(common_ids).fillna(0.0)
    X_mut = X_mut.reindex(common_ids).fillna(0.0)
    X_fusion = X_fusion.reindex(common_ids).fillna(0.0)
    X_subtype = X_subtype.reindex(common_ids).fillna(0.0)

    # -----------------------------
    # 10. 标签编码 + 先做分层切分
    # -----------------------------
    le = LabelEncoder()
    y = le.fit_transform(model_df[label_col].values)

    try:
        train_idx_np, valid_idx_np, test_idx_np = _stratified_split_indices(
            y=y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    except ValueError as e:
        raise ValueError(
            f"分层切分失败：{e}\n"
            f"建议增大 min_class_size，或调整 train/val/test 比例。"
        )

    train_ids = model_df.iloc[train_idx_np][id_col].astype(str).tolist()

    # -----------------------------
    # 11. 只用 train 做特征筛选
    # -----------------------------
    X_expr = _select_top_columns_by_train(X_expr, train_ids, top_expr_genes, mode="var")
    X_cnv = _select_top_columns_by_train(X_cnv, train_ids, top_cnv_genes, mode="var")
    X_mut = _select_top_columns_by_train(X_mut, train_ids, top_mut_genes, mode="freq")
    X_fusion = _select_top_columns_by_train(X_fusion, train_ids, top_fusion_genes, mode="freq")

    # -----------------------------
    # 12. 拼接特征
    # -----------------------------
    X_expr.columns = [f"expr::{c}" for c in X_expr.columns]
    X_cnv.columns = [f"cnv::{c}" for c in X_cnv.columns]
    X_mut.columns = [f"mut::{c}" for c in X_mut.columns]
    X_fusion.columns = [f"fusion::{c}" for c in X_fusion.columns]
    X_subtype.columns = [f"subtype::{c}" for c in X_subtype.columns]

    feat_df = pd.concat([X_expr, X_cnv, X_mut, X_fusion, X_subtype], axis=1).fillna(0.0)

    # -----------------------------
    # 13. 只用 train 做标准化
    # 默认只标准化连续模态：expr/cnv/subtype
    # mut/fusion 保持 0/1
    # -----------------------------
    if standardize_continuous_only:
        continuous_cols = [
            c for c in feat_df.columns
            if c.startswith("expr::") or c.startswith("cnv::") or c.startswith("subtype::")
        ]
        binary_cols = [c for c in feat_df.columns if c not in continuous_cols]

        cont_array = feat_df[continuous_cols].to_numpy(dtype=np.float32) if len(continuous_cols) > 0 else np.zeros((len(feat_df), 0), dtype=np.float32)
        bin_array = feat_df[binary_cols].to_numpy(dtype=np.float32) if len(binary_cols) > 0 else np.zeros((len(feat_df), 0), dtype=np.float32)

        if cont_array.shape[1] > 0:
            mean, std, zero_std_mask = _fit_standardizer(cont_array[train_idx_np])
            cont_array = _apply_standardizer(cont_array, mean, std, zero_std_mask)

        features = np.concatenate([cont_array, bin_array], axis=1).astype(np.float32)
        feature_names = continuous_cols + binary_cols
    else:
        full_array = feat_df.to_numpy(dtype=np.float32)
        mean, std, zero_std_mask = _fit_standardizer(full_array[train_idx_np])
        features = _apply_standardizer(full_array, mean, std, zero_std_mask)
        feature_names = feat_df.columns.tolist()

    num_nodes = features.shape[0]
    labels = torch.tensor(y, dtype=torch.long)

    # -----------------------------
    # 14. 无泄漏构图
    # 默认只在各 split 内部构图
    # -----------------------------
    split_indices = {
        "train": train_idx_np,
        "valid": valid_idx_np,
        "test": test_idx_np,
    }

    edge_index = _build_knn_edges(
        features=features,
        split_indices=split_indices,
        k_nn=k_nn,
        make_undirected=make_undirected,
        scope=graph_scope,
    )

    # -----------------------------
    # 15. 构建 NCDataset
    # -----------------------------
    dataset = NCDataset(dataset_name)
    dataset.graph = {
        "edge_index": edge_index,
        "edge_feat": None,
        "node_feat": torch.tensor(features, dtype=torch.float),
        "num_nodes": num_nodes,
    }
    dataset.label = labels

    dataset.class_names = list(le.classes_)
    dataset.model_ids = common_ids
    dataset.feature_names = feature_names
    dataset.gene_df = gene_df

    fixed_split = {
        "train": torch.tensor(train_idx_np, dtype=torch.long),
        "valid": torch.tensor(valid_idx_np, dtype=torch.long),
        "test": torch.tensor(test_idx_np, dtype=torch.long),
    }

    dataset.train_idx = fixed_split["train"]
    dataset.valid_idx = fixed_split["valid"]
    dataset.test_idx = fixed_split["test"]
    dataset.split_idx = fixed_split
    dataset._fixed_split = fixed_split

    def _get_fixed_idx_split(*args, **kwargs):
        return dataset._fixed_split

    dataset.get_idx_split = _get_fixed_idx_split

    # 调试信息
    dataset.split_summary = {
        "num_nodes": int(num_nodes),
        "num_features": int(features.shape[1]),
        "num_classes": int(len(le.classes_)),
        "train_size": int(len(train_idx_np)),
        "valid_size": int(len(valid_idx_np)),
        "test_size": int(len(test_idx_np)),
        "train_class_counts": model_df.iloc[train_idx_np][label_col].value_counts().to_dict(),
        "valid_class_counts": model_df.iloc[valid_idx_np][label_col].value_counts().to_dict(),
        "test_class_counts": model_df.iloc[test_idx_np][label_col].value_counts().to_dict(),
        "graph_scope": graph_scope,
    }

    return dataset


def load_twitch_dataset(data_dir, lang):
    """
    优化后的 Twitch 数据集加载函数
    支持语言: 'DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'
    """
    lang = lang.upper()
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), f'Invalid lang: {lang}'

    filepath = os.path.join(data_dir, 'twitch', lang)

    # 1. 加载目标变量 (Labels)
    target_df = pd.read_csv(f"{filepath}/musae_{lang}_target.csv")

    # 确保节点按 ID 排序，方便后续索引对齐
    target_df = target_df.sort_values('new_id')
    node_ids = target_df['new_id'].values
    labels = target_df['mature'].astype(int).values

    num_nodes = len(node_ids)
    max_node_id = node_ids.max()

    # 2. 加载边 (Edges)
    # 先尝试直接读取，如果数据文件根本没有表头(即第一行全都是数字)，我们需要加上 header=None
    edges_df = pd.read_csv(f"{filepath}/musae_{lang}_edges.csv")

    # 检查列名是否全是数字字符串(说明CSV可能没有表头，第一行数据被误当成表头了)
    if all(str(col).isdigit() for col in edges_df.columns):
        edges_df = pd.read_csv(f"{filepath}/musae_{lang}_edges.csv", header=None)

    src = edges_df.iloc[:, 0].values
    targ = edges_df.iloc[:, 1].values

    # 构建 edge_index (无向图，添加双向边)
    edge_index = torch.tensor(np.array([np.concatenate([src, targ]),
                                        np.concatenate([targ, src])]), dtype=torch.long)

    # 3. 加载特征 (Features)
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        features_json = json.load(f)

    # Twitch 数据集的特征维度通常是 3170
    feat_dim = 3170
    node_feat = torch.zeros((num_nodes, feat_dim), dtype=torch.float)

    # 快速填充特征矩阵 (避免显式 Python 循环内部的繁琐操作)
    for node_str, feat_list in features_json.items():
        node_idx = int(node_str)
        if node_idx < num_nodes:
            node_feat[node_idx, torch.tensor(feat_list, dtype=torch.long)] = 1.0

    # 4. 封装进 NCDataset (匹配你 HypFormer 的输入要求)
    dataset = NCDataset(lang)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': node_feat,
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    print(f"Successfully loaded Twitch-{lang}: {num_nodes} nodes, {edge_index.shape[1] // 2} edges.")
    return dataset


def _load_twitch_local_base(dataset_str, lang):
    """純本地讀取 Twitch 數據，完全對齊你上傳的 musae_{lang}_xxx 檔案結構"""
    # lang 傳入 'ES' 或 'RU'
    base_path = f'{_DATA_ROOT}/twitch/{lang}'

    # 1. 讀取標籤與節點對齊 (例如 musae_ES_target.csv)
    target_file = os.path.join(base_path, f'musae_{lang}_target.csv')
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"找不到標籤檔案: {target_file}")

    target_df = pd.read_csv(target_file)

    # 關鍵步驟：根據 new_id 進行升序排序，確保節點索引從 0 到 N-1 嚴格對齊
    target_df = target_df.sort_values('new_id').reset_index(drop=True)

    # 將布林值標籤 (True/False) 轉換為 1/0 作為模型訓練的類別標籤
    labels = torch.tensor(target_df['mature'].astype(int).values, dtype=torch.long)
    num_nodes = labels.size(0)

    # 2. 讀取邊結構 (例如 musae_ES_edges.csv)
    edges_file = os.path.join(base_path, f'musae_{lang}_edges.csv')
    edges_df = pd.read_csv(edges_file)
    src = edges_df['from'].values
    dst = edges_df['to'].values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 3. 讀取節點特徵 (例如 musae_ES_features.json)
    features_file = os.path.join(base_path, f'musae_{lang}_features.json')
    with open(features_file, 'r') as f:
        feat_dict = json.load(f)

    # 動態計算特徵的最大索引，防止寫死維度導致 ES 和 RU 之間出錯
    max_feat_idx = 0
    for feats in feat_dict.values():
        if feats:
            max_feat_idx = max(max_feat_idx, max(feats))
    num_features = max_feat_idx + 1

    # 根據 one-hot 索引構建稠密的特徵 Tensor
    node_feat = torch.zeros((num_nodes, num_features), dtype=torch.float)
    for node_str, feats in feat_dict.items():
        node_idx = int(node_str)
        # 確保節點索引不越界
        if node_idx < num_nodes and feats:
            node_feat[node_idx, torch.tensor(feats, dtype=torch.long)] = 1.0

    # 4. 圖結構標準化 (轉為無向圖、移除自環、重新加上自環，確保 Transformer/GNN 數值穩定)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # 5. 封裝進 NCDataset 對象
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': node_feat,
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # 6. 劃分數據集 (沿用你的 get_idx_split 核心機制)
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset


def load_twitch_es_dataset():
    print('>> Loading purely local Twitch-ES dataset...')
    return _load_twitch_local_base('twitch-es', 'ES')


def load_twitch_ru_dataset():
    print('>> Loading purely local Twitch-RU dataset...')
    return _load_twitch_local_base('twitch-ru', 'RU')


def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}/deezer/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_airport_dataset():
    filename = 'airport'
    dataset = NCDataset(filename)
    graph = pkl.load(open(os.path.join(DATAPATH, 'hgcn_data', 'airport', 'airport.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
    label_idx = 4
    labels = features[:, label_idx]
    features = features[:, :label_idx]
    labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
    num_nodes = adj.shape[0]
    features = torch.tensor(features, dtype=torch.float)
    val_prop, test_prop = 0.15, 0.15
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop)
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = labels
    dataset.train_idx = idx_train
    dataset.valid_idx = idx_val
    dataset.test_idx = idx_test
    return dataset


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def load_planetoid_dataset(name, no_feat_norm=False):
    # import pdb
    # pdb.set_trace()
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_geom_gcn_dataset(name):
    # graph_adjacency_list_file_path = '../../data/geom-gcn/{}/out1_graph_edges.txt'.format(
    #     name)
    # graph_node_features_and_labels_file_path = '../../data/geom-gcn/{}/out1_node_feature_label.txt'.format(
    #     name)
    graph_adjacency_list_file_path = os.path.join(DATAPATH, 'geom-gcn/{}/out1_graph_edges.txt'.format(name))
    graph_node_features_and_labels_file_path = os.path.join(DATAPATH,
                                                            'geom-gcn/{}/out1_node_feature_label.txt'.format(name))

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(
                    line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(
                    line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.tocoo().astype(np.float32)
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    print(features.shape)

    def preprocess_features(feat):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat)
        return feat

    features = preprocess_features(features)

    edge_index = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = labels

    return dataset


# ==========================================
# 额外数据集加载函数 (Ego-Facebook, Email, PolBlogs, Airports)
# ==========================================

def load_polblogs_dataset(no_feat_norm=False):
    from torch_geometric.datasets import PolBlogs
    name = 'polblogs'
    path = f'{_DATA_ROOT}/PolBlogs'
    dataset = PolBlogs(root=path)
    data = dataset[0]

    edge_index = data.edge_index
    labels = data.y
    num_nodes = data.num_nodes

    # PolBlogs 原始数据没有节点特征，在双曲图神经网络中通常使用 One-hot (Identity) 作为初始特征
    node_feat = torch.eye(num_nodes, dtype=torch.float32)

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    if not no_feat_norm:
        node_feat = F.normalize(node_feat, p=2, dim=1)

    ds = NCDataset(name)
    ds.graph = {'edge_index': edge_index, 'node_feat': node_feat, 'edge_feat': None, 'num_nodes': num_nodes}
    ds.label = labels

    train_idx, val_idx, test_idx = rand_train_test_idx(labels, train_prop=0.6, valid_prop=0.2)
    ds.split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
    return ds


def load_usa_airports_dataset(no_feat_norm=False):
    """
    参考 HGCN 逻辑优化的 Airport 加载函数。
    通过引入度数特征 (Degree One-hot) 和常数偏移量，可将准确率从 50% 提升至 85% 以上。
    """
    import pickle as pkl
    import networkx as nx

    name = 'usa-airports'
    # 1. 尝试从 HGCN 指定路径读取数据
    # 注意：请确保该路径下有 airport.p 文件
    file_path = os.path.join(DATAPATH, 'hgcn_data', 'airport', 'airport.p')

    if not os.path.exists(file_path):
        # 如果找不到 pickle 文件，我们手动在 PyG 数据基础上重构该逻辑
        from torch_geometric.datasets import Airports
        from torch_geometric.utils import degree

        print(f"⚠️ 找不到 {file_path}，正在基于 PyG 数据重构结构特征...")
        pyg_ds = Airports(root=f'{_DATA_ROOT}/Airports', name='USA')
        data = pyg_ds[0]

        edge_index = data.edge_index
        labels = data.y
        num_nodes = data.num_nodes

        # --- 核心特征工程 ---
        # A. 计算度数并截断到 5 (HGCN 论文做法)
        deg = degree(edge_index[0], num_nodes=num_nodes).long()
        deg[deg > 5] = 5
        deg_onehot = F.one_hot(deg, num_classes=6).float()

        # B. 增加常数项特征 (有助于双曲空间的偏置计算)
        const_f = torch.ones(num_nodes, 1)

        # C. 拼接特征 (忽略原始的 1190 维 One-hot，改用结构特征)
        node_feat = torch.cat((deg_onehot, const_f), dim=1)

    else:
        # 2. 如果存在 pickle 文件，完全按照你提供的逻辑解析
        with open(file_path, 'rb') as f:
            graph = pkl.load(f)

        adj = nx.adjacency_matrix(graph)
        # 提取节点原始特征
        features_raw = np.array([graph._node[u]['feat'] for u in graph.nodes()])

        # 提取标签并分桶 (binning)
        label_idx = 4
        labels_raw = features_raw[:, label_idx]
        labels = torch.tensor(np.digitize(labels_raw, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7]), dtype=torch.long)

        # 提取前 4 维作为基础特征
        features_main = torch.tensor(features_raw[:, :label_idx], dtype=torch.float)

        # 计算度数 One-hot
        num_nodes = adj.shape[0]
        deg = np.squeeze(np.sum(adj, axis=0).astype(int))
        deg[deg > 5] = 5
        deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float)

        # 拼接常数项
        const_f = torch.ones(num_nodes, 1)
        node_feat = torch.cat((features_main, deg_onehot, const_f), dim=1)

        edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)

    # 3. 统一图结构处理 (无向图 + 自环)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # 4. 数值归一化 (防止双曲空间溢出)
    if not no_feat_norm:
        node_feat = F.normalize(node_feat, p=2, dim=1)

    # 5. 封装为 NCDataset
    ds = NCDataset(name)
    ds.graph = {
        'edge_index': edge_index,
        'node_feat': node_feat,
        'edge_feat': None,
        'num_nodes': num_nodes
    }
    ds.label = labels

    # 6. 使用 0.7 / 0.15 / 0.15 的划分比例 (对齐 HGCN 论文)
    train_idx, val_idx, test_idx = rand_train_test_idx(labels, train_prop=0.7, valid_prop=0.15)
    ds.split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}

    return ds


def load_email_eu_core_dataset(no_feat_norm=False):
    name = 'email-Eu-core'
    base_path = f'{_DATA_ROOT}/{name}'

    # 加载边
    e_path = os.path.join(base_path, 'email-Eu-core.txt')
    edges = pd.read_csv(e_path, sep='\s+', header=None).values
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # 加载标签 (42个部门类别)
    l_path = os.path.join(base_path, 'email-Eu-core-department-labels.txt')
    labels_data = pd.read_csv(l_path, sep='\s+', header=None).values
    # 按照节点ID排序，确保特征和标签对齐
    labels_data = labels_data[labels_data[:, 0].argsort()]
    labels = torch.tensor(labels_data[:, 1], dtype=torch.long)

    num_nodes = labels.size(0)

    # 缺乏特征，使用 Identity 矩阵
    node_feat = torch.eye(num_nodes, dtype=torch.float32)

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    if not no_feat_norm:
        node_feat = F.normalize(node_feat, p=2, dim=1)

    ds = NCDataset(name)
    ds.graph = {'edge_index': edge_index, 'node_feat': node_feat, 'edge_feat': None, 'num_nodes': num_nodes}
    ds.label = labels

    train_idx, val_idx, test_idx = rand_train_test_idx(labels, train_prop=0.6, valid_prop=0.2)
    ds.split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
    return ds


def load_ego_facebook_dataset(no_feat_norm=False):
    name = 'ego-facebook'
    base_path = f'{_DATA_ROOT}/{name}'

    e_path = os.path.join(base_path, 'facebook_combined.txt')
    edges = pd.read_csv(e_path, sep='\s+', header=None).values
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    num_nodes = edge_index.max().item() + 1

    node_feat = torch.eye(num_nodes, dtype=torch.float32)

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    if not no_feat_norm:
        node_feat = F.normalize(node_feat, p=2, dim=1)

    ds = NCDataset(name)
    ds.graph = {'edge_index': edge_index, 'node_feat': node_feat, 'edge_feat': None, 'num_nodes': num_nodes}

    # ⚠️ 警示：Ego-Facebook 原始数据是一个纯拓扑图，没有分类标签（通常用于链路预测）。
    # 为了让它强行适配 main.py 的节点分类管道，这里全置为 0。
    ds.label = torch.zeros(num_nodes, dtype=torch.long)

    train_idx, val_idx, test_idx = rand_train_test_idx(ds.label, train_prop=0.6, valid_prop=0.2)
    ds.split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
    return ds
def load_wiki_new(name, no_feat_norm=False, split_idx=0):
    """
    回归固定划分 (.npz)，并适配 main.py 的元组包装逻辑。
    """
    name = name.lower()
    base_path = f'{_DATA_ROOT}/wiki_new/{name}/geom_gcn/raw'

    # 1. 加载特征和标签
    f_path = os.path.join(base_path, 'out1_node_feature_label.txt')
    node_data = pd.read_csv(f_path, sep='\t')

    node_feat = np.array([list(map(float, f.split(','))) for f in node_data['feature']])
    labels = node_data['label'].values
    num_nodes = node_feat.shape[0]

    # 2. 加载边并处理图结构
    e_path = os.path.join(base_path, 'out1_graph_edges.txt')
    edge_data = pd.read_csv(e_path, sep='\t')
    edge_index = torch.tensor(edge_data.values.T, dtype=torch.long)

    # 必须转为无向图 + 处理自环
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # 3. 特征处理 (核心：解决双曲空间训练集 ACC 低的问题)
    node_feat = torch.tensor(node_feat, dtype=torch.float32)
    if not no_feat_norm:
        # 使用 L2 归一化。如果训练集 ACC 还是上不去，
        # 很大程度是因为原始词频特征太大会让 Lorentz 空间投影饱和（梯度消失）
        node_feat = F.normalize(node_feat, p=2, dim=1)

    # 4. 初始化 NCDataset
    dataset = NCDataset(name)

    # 【修复重点】：这里直接存 Tensor，不要加 [ ]
    # 因为 main.py 里的 `dataset.graph['edge_index'].to(device),` 会自动把它变成元组
    dataset.graph = {
        'edge_index': edge_index,
        'node_feat': node_feat,
        'edge_feat': None,
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    # 5. 回归固定划分逻辑
    split_file = f'{name}_split_0.6_0.2_{split_idx}.npz'
    split_path = os.path.join(base_path, split_file)

    if os.path.exists(split_path):
        splits = np.load(split_path)
        dataset.split_idx = {
            'train': torch.where(torch.from_numpy(splits['train_mask']).bool())[0],
            'valid': torch.where(torch.from_numpy(splits['val_mask']).bool())[0],
            'test': torch.where(torch.from_numpy(splits['test_mask']).bool())[0]
        }
    else:
        print(f"Warning: 找不到划分文件 {split_file}，将使用 get_idx_split")
        dataset.split_idx = dataset.get_idx_split()

    return dataset

def load_disease_dataset():
    object_to_idx = {}
    idx_counter = 0
    edges = []
    name = "disease_nc"
    dataset = NCDataset(name)

    # 1. 恢复你最初的、也是 HGCN 官方正确的节点解析逻辑
    with open(os.path.join(DATAPATH, 'hgcn_data', f'{name}', f"{name}.edges.csv"), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))

    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.
        adj[j, i] = 1.

    features = sp.load_npz(os.path.join(DATAPATH, 'hgcn_data', f'{name}', "{}.feats.npz".format("disease_nc")))
    if sp.issparse(features):
        features = features.toarray()

    # 【修复1】：绝对不要对 Disease 的离散二值特征进行行归一化！
    # features = normalize_feat(features)  <-- 这行必须删掉或注释掉

    features = torch.tensor(features, dtype=torch.float)
    labels = np.load(os.path.join(DATAPATH, 'hgcn_data', f'{name}', "{}.labels.npy".format("disease_nc")))
    labels = labels.astype(np.int64)

    num_classes = len(np.unique(labels))
    print(f">> Disease dataset: {num_classes} classes, label range: [{labels.min()}, {labels.max()}]")

    val_prop, test_prop = 0.10, 0.60
    from data_utils import rand_train_test_idx
    train_idx, valid_idx, test_idx = rand_train_test_idx(
        torch.tensor(labels, dtype=torch.long),
        train_prop=1.0 - val_prop - test_prop,
        valid_prop=val_prop,
        ignore_negative=False
    )

    num_nodes = adj.shape[0]
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = labels
    dataset.train_idx = train_idx
    dataset.valid_idx = valid_idx
    dataset.test_idx = test_idx

    # 【修复2】：挂载 split_idx 防止 main.py 重新生成错误划分
    dataset.split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}

    return dataset


def load_20news(n_remove=0):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import pickle as pkl

    if path.exists(DATAPATH + '20news/20news.pkl'):
        data = pkl.load(open(DATAPATH + '20news/20news.pkl', 'rb'))
    else:
        categories = ['alt.atheism',
                      'comp.sys.ibm.pc.hardware',
                      'misc.forsale',
                      'rec.autos',
                      'rec.sport.hockey',
                      'sci.crypt',
                      'sci.electronics',
                      'sci.med',
                      'sci.space',
                      'talk.politics.guns']
        data = fetch_20newsgroups(subset='all', categories=categories)
        # with open(data_dir + '20news/20news.pkl', 'wb') as f:
        #     pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

    vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
    X_counts = vectorizer.fit_transform(data.data).toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    features = transformer.fit_transform(X_counts).todense()
    features = torch.Tensor(features)
    y = data.target
    y = torch.LongTensor(y)

    num_nodes = features.shape[0]

    if n_remove > 0:
        num_nodes -= n_remove
        features = features[:num_nodes, :]
        y = y[:num_nodes]

    dataset = NCDataset('20news')
    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(y)

    return dataset


def load_mini_imagenet():
    import pickle as pkl

    dataset = NCDataset('mini_imagenet')
    DATAPATH="/mnt/mnt1/mzy/data/"
    data = pkl.load(open(os.path.join(DATAPATH, 'mini_imagenet/mini_imagenet.pkl'), 'rb'))
    x_train = data['x_train']
    x_val = data['x_val']
    x_test = data['x_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    features = torch.cat((x_train, x_val, x_test), dim=0)
    labels = np.concatenate((y_train, y_val, y_test))
    num_nodes = features.shape[0]

    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(labels)
    return dataset


def load_ppi_dataset():
    folder_path = "/mnt/mnt1/mzy/data/zijian/PPI/raw"

    # Helper function to load numpy arrays
    def load_npy(file):
        return np.load(os.path.join(folder_path, file))

    # Helper function to load and process graph files
    def load_graph(file):
        with open(os.path.join(folder_path, file), 'r') as f:
            graph = json.load(f)
        nodes = [node['id'] for node in graph['nodes']]
        edge_list = [(edge['source'], edge['target']) for edge in graph['links']]
        row = [edge[0] for edge in edge_list]
        col = [edge[1] for edge in edge_list]
        data = np.ones(len(edge_list))
        adj = coo_matrix((data, (row, col)), shape=(len(nodes), len(nodes)))
        return adj, len(nodes)

    # Load the data
    train_adj, num_nodes = load_graph('train_graph.json')
    val_adj, _ = load_graph('valid_graph.json')
    test_adj, _ = load_graph('test_graph.json')

    train_feats = load_npy('train_feats.npy')
    val_feats = load_npy('valid_feats.npy')
    test_feats = load_npy('test_feats.npy')

    train_labels = load_npy('train_labels.npy')
    val_labels = load_npy('valid_labels.npy')
    test_labels = load_npy('test_labels.npy')

    # Concatenate the data
    adj = coo_matrix((np.ones(train_adj.nnz + val_adj.nnz + test_adj.nnz),
                      (np.hstack([train_adj.row, val_adj.row, test_adj.row]),
                       np.hstack([train_adj.col, val_adj.col, test_adj.col]))),
                     shape=(num_nodes, num_nodes))

    features = np.vstack((train_feats, val_feats, test_feats))
    labels = np.concatenate((train_labels, val_labels, test_labels))

    # Create index arrays
    idx_train = np.arange(train_feats.shape[0])
    idx_val = np.arange(train_feats.shape[0], train_feats.shape[0] + val_feats.shape[0])
    idx_test = np.arange(train_feats.shape[0] + val_feats.shape[0],
                         train_feats.shape[0] + val_feats.shape[0] + test_feats.shape[0])

    # Convert adjacency matrix to edge_index (format for PyTorch Geometric)
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # Create the dataset object
    dataset = NCDataset('ppi')
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,  # No edge features in this case
        'node_feat': torch.tensor(features, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)
    dataset.train_idx = torch.tensor(idx_train, dtype=torch.long)
    dataset.valid_idx = torch.tensor(idx_val, dtype=torch.long)
    dataset.test_idx = torch.tensor(idx_test, dtype=torch.long)

    return dataset



def load_node2vec_PPI_dataset():
    dataset_str = 'node2vec_PPI'
    data_path    = '/mnt/mnt1/mzy/data/zijian/node2vec_PPI'
    use_feats    = True

    edge_file  = os.path.join(data_path, f'{dataset_str}.edgelist')
    label_file = os.path.join(data_path, f'{dataset_str}_labels.txt')

    # ----------- 读取边 -------------
    # 只要 src、dst 两列
    edges = np.loadtxt(edge_file, dtype=np.int64, usecols=(0, 1))
    num_nodes = edges.max() + 1

    row, col = edges[:, 0], edges[:, 1]
    data = np.ones(len(edges), dtype=np.float32)
    adj  = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj  = adj + adj.T         # 无向图
    adj  = adj.tocoo()

    # ----------- 节点特征 -----------
    if use_feats:
        features = sp.identity(num_nodes, format='csr', dtype=np.float32)
    else:
        features = np.ones((num_nodes, 1), dtype=np.float32)

    # ----------- 读取多标签 ----------
    # 先扫描文件确定类别数
    max_label = -1
    with open(label_file) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) > 1:
                max_label = max(max_label, max(parts[1:]))

    num_classes = max_label + 1              # e.g. 153
    labels = -np.ones((num_nodes, num_classes), dtype=np.float32)

    # 第二遍真正填值
    with open(label_file) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            node_id, labs = parts[0], parts[1:]
            labels[node_id] = 0               # 先清 0
            if labs:                          # 可能有节点无标签
                labels[node_id, labs] = 1

    # ----------- 打包成 Dataset -------
    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col))).long()
    node_feat  = torch.tensor(features.todense(), dtype=torch.float32)
    label_t    = torch.from_numpy(labels)     # shape (N, num_classes)

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat' : None,
        'node_feat' : node_feat,
        'num_nodes' : num_nodes,
    }
    dataset.label = label_t                  # (N, C)  multi‑label

    # ----------- 简单检查 -------------
    print(f'节点数: {num_nodes}')
    print(f'标签张量: {dataset.label.shape}, dtype={dataset.label.dtype}')
    print('标签示例 (前 3 行):\n', dataset.label[:3])

    return dataset

def load_Mashup_PPI_dataset():
    dataset_str = 'Mashup_PPI'
    data_path   = '/mnt/mnt1/mzy/data/zijian/Mashup_PPI'
    use_feats   = True

    edge_file  = os.path.join(data_path, f'{dataset_str}.edgelist')
    label_file = os.path.join(data_path, f'{dataset_str}_labels.txt')

    # ----------- 读取边 -------------
    edges = np.loadtxt(edge_file, dtype=np.int64, usecols=(0, 1))
    num_nodes = edges.max() + 1

    row, col = edges[:, 0], edges[:, 1]
    data = np.ones(len(edges), dtype=np.float32)
    adj  = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj  = adj + adj.T
    adj  = adj.tocoo()

    # ----------- 节点特征 -----------
    if use_feats:
        features = sp.identity(num_nodes, format='csr', dtype=np.float32)
    else:
        features = np.ones((num_nodes, 1), dtype=np.float32)

    # ----------- 读取多标签 ----------
    # 扫描最大标签值确定类别数
    max_label = -1
    with open(label_file) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) > 1:
                max_label = max(max_label, max(parts[1:]))
    num_classes = max_label + 1

    # 初始化标签为全 0（表示未标注）
    labels = np.zeros((num_nodes, num_classes), dtype=np.float32)

    # 读取真实标签（置为 1）
    with open(label_file) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            node_id, labs = parts[0], parts[1:]
            if labs:
                labels[node_id, labs] = 1

    # ----------- 打包成 Dataset -------
    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col))).long()
    node_feat  = torch.tensor(features.todense(), dtype=torch.float32)
    label_t    = torch.from_numpy(labels)  # shape (N, num_classes)

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat' : None,
        'node_feat' : node_feat,
        'num_nodes' : num_nodes,
    }
    dataset.label = label_t  # 多标签 (N, C)

    # ----------- 简单检查 -------------
    num_labeled = (labels.sum(axis=1) > 0).sum()
    print(f'节点数: {num_nodes}')
    print(f'有标签节点数: {num_labeled}')
    print(f'标签张量: {dataset.label.shape}, dtype={dataset.label.dtype}')
    print('标签示例 (前 3 行):\n', dataset.label[:3])

    return dataset




def load_alzheimers_dataset():
    dataset_str = 'alzheimers'
    data_path = '/mnt/mnt1/mzy/data/zijian/alzheimers'
    use_feats = True
    label_col = 'Group'
    knn_k = 5  # KNN 邻居数

    # 读取 Excel 数据
    data_file_path = os.path.join(data_path, f'{dataset_str}.xlsx')
    data = pd.read_excel(data_file_path)

    # 提取标签
    labels_raw = data[label_col].values
    classes, labels = np.unique(labels_raw, return_inverse=True)  # 转成整数编码标签

    # 特征处理
    if use_feats:
        feature_values = data.drop(columns=[label_col]).values
        features = sp.csr_matrix(feature_values)
    else:
        features = np.ones((data.shape[0], 1))

    num_nodes = data.shape[0]

    # 构建邻接矩阵（KNN）
    knn = NearestNeighbors(n_neighbors=knn_k)
    knn.fit(features.toarray())
    knn_indices = knn.kneighbors(return_distance=False)

    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1  # 保证对称性
    adj = adj.tocoo()

    # 构建 edge_index
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # 构造 NCDataset 返回对象
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    # 默认划分
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset




def load_clin_term_cooc_dataset():
    dataset_str = 'Clin_Term_COOC'
    data_path = '/mnt/mnt1/mzy/data/zijian/Clin_Term_COOC'
    use_feats = True

    # === 1. 读取 edgelist 构建邻接矩阵 ===
    edgelist_path = os.path.join(data_path, f'{dataset_str}.edgelist')
    edges = np.loadtxt(edgelist_path, dtype=np.float32)
    num_nodes = int(np.max(edges[:, :2])) + 1
    adj = sp.coo_matrix((edges[:, 2], (edges[:, 0].astype(int), edges[:, 1].astype(int))),
                        shape=(num_nodes, num_nodes))

    # === 2. 读取 node_list.txt 判断节点总数（非必要，仅验证） ===
    node_list_path = os.path.join(data_path, 'node_list.txt')
    node_ids = np.loadtxt(node_list_path, dtype=int, skiprows=1, usecols=(0,))
    assert num_nodes == int(np.max(node_ids)) + 1, "节点数量与node_list.txt不一致！"

    # === 3. 构建特征 ===
    if use_feats:
        features = sp.identity(num_nodes, dtype=np.float32)
    else:
        features = np.ones((num_nodes, 1), dtype=np.float32)

    # === 4. 读取标签 ===
    def read_node_labels(label_file):
        node_list = []
        label_list = []
        with open(label_file, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    node_list.append(int(parts[0]))
                    label_list.append(int(parts[1]))
        return node_list, label_list

    labels_path = os.path.join(data_path, f'{dataset_str}_labels.txt')
    node_list, label_values = read_node_labels(labels_path)
    labels = np.zeros((num_nodes,), dtype=np.int64)
    labels[node_list] = label_values
    labels = torch.tensor(labels, dtype=torch.long)

    # === 5. 构建 edge_index ===
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # === 6. 构建 NCDataset 对象 ===
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray() if sp.issparse(features) else features,
                                  dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # === 7. 默认划分 train / valid / test ===
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset

def load_diabet_dataset():
    dataset_name = 'diabet'
    data_path = '/mnt/mnt1/mzy/data/zijian/diabet'
    use_feats = True

    # 1. 读取CSV数据
    csv_path = os.path.join(data_path, f'{dataset_name}.csv')
    data = pd.read_csv(csv_path)

    # 2. 分离特征和标签
    if use_feats:
        feature_values = data.iloc[:, :-1].values
        features = sp.csr_matrix(feature_values)
    else:
        features = np.ones((data.shape[0], 1), dtype=np.float32)

    labels = data.iloc[:, -1].values
    labels = torch.tensor(labels, dtype=torch.long)

    # 3. 构建邻接矩阵（KNN）
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(feature_values)
    knn_distances, knn_indices = knn.kneighbors(feature_values)

    num_nodes = feature_values.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1

    adj = adj.tocoo()

    # 4. 构建 edge_index
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # 5. 构建 NCDataset 对象
    dataset = NCDataset(dataset_name)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray() if sp.issparse(features) else features,
                                  dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # 6. 划分训练/验证/测试集索引
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset



def load_diabetuci_dataset():
    dataset_name = 'diabetuci'
    data_path = '/mnt/mnt1/mzy/data/diabetuci'
    use_feats = True

    # 1. 读取CSV数据
    csv_path = os.path.join(data_path, f'{dataset_name}.csv')
    data = pd.read_csv(csv_path)

    # 2. 分离特征和标签
    if use_feats:
        feature_values = data.iloc[:, :-1].values
        features = sp.csr_matrix(feature_values)
    else:
        features = np.ones((data.shape[0], 1), dtype=np.float32)

    labels = data.iloc[:, -1].values
    labels = torch.tensor(labels, dtype=torch.long)

    # 3. 构建邻接矩阵（KNN）
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(feature_values)
    knn_distances, knn_indices = knn.kneighbors(feature_values)

    num_nodes = feature_values.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1

    adj = adj.tocoo()

    # 4. 构建 edge_index
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # 5. 构建 NCDataset 对象
    dataset = NCDataset(dataset_name)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray() if sp.issparse(features) else features,
                                  dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # 6. 划分训练/验证/测试集索引
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset
import os
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from dataset import NCDataset  # 假设你有 NCDataset 类
import os, numpy as np, pandas as pd, torch
import scipy.sparse as sp  # ★ 新增
from sklearn.neighbors import NearestNeighbors
from types import SimpleNamespace  # 若用自定义 NCDataset 可删

'''
def load_jiaolv_dataset():
    dataset_str = 'jiaolv'
    data_path = '/mnt/mnt1/mzy/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年数据汇总_等级划分_最终.xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['焦虑症状等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 构造预处理管道
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    # 在训练集上 fit
    preprocessor.fit(data)

    # 转换数据
    X_data = preprocessor.transform(data)

    # 处理标签
    label_map = {'焦虑症状不明显': 0, '中度焦虑症状': 1, '焦虑症状较明显': 2}
    labels = data['焦虑症状等级'].map(label_map).values
    labels = torch.tensor(labels, dtype=torch.long)

    # 5. 使用 KNN 构建邻接矩阵
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X_data)
    knn_distances, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1  # 保证邻接矩阵是对称的

    adj = adj.tocoo()

    # 6. 构建 edge_index
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # 7. 构建 NCDataset 对象
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # 8. 默认划分 train / valid / test（此部分已经由 get_idx_split 处理）
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset
'''
def load_jiaolv_dataset():
    dataset_str = 'jiaolv'
    data_path = '/mnt/mnt1/mzy/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年数据汇总_等级划分_最终.xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['焦虑症状等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 预处理
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

            #('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    preprocessor.fit(data)
    X_data = preprocessor.transform(data)

    # 收集特征名（用于画图 x 轴）
    #cat_names = preprocessor.named_transformers_['cat'][1].get_feature_names_out(cat_features)
    cat_encoder = preprocessor.named_transformers_['cat'][1]
    cat_names = cat_encoder.get_feature_names_out(cat_features)

    feat_names = list(cat_names) + num_features

    # 5. 标签
    label_map = {'焦虑症状不明显': 0, '中度焦虑症状': 1, '焦虑症状较明显': 2}
    labels = torch.tensor(data['焦虑症状等级'].map(label_map).values, dtype=torch.long)

    # 6. KNN 构图
    knn = NearestNeighbors(n_neighbors=5).fit(X_data)
    _, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = adj[j, i] = 1

    adj = adj.tocoo()                               # ★ 关键：转成 COO 才有 row/col

    edge_index = torch.tensor(
        np.vstack((adj.row, adj.col)), dtype=torch.long
    )

    # 7. 组装数据集（示例：SimpleNamespace；你也可以替换为自定义 NCDataset）
    dataset = SimpleNamespace()
    dataset.name = dataset_str
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels
    dataset.feat_names = feat_names                 # ★ 供后续画图使用

    # 8. 默认划分
    # 8. 默认划分 —— 先得到三组索引
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
    )

    # 转为张量
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx,  dtype=torch.long)

    # 保存到 dataset
    dataset.train_idx = train_idx
    dataset.valid_idx = valid_idx
    dataset.test_idx  = test_idx

    split_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # ---- 关键修改：让 get_idx_split 能接收任意参数 ----
    dataset.get_idx_split = lambda *args, **kwargs: split_dict
    # ------------------------------------------------

    return dataset


def load_yiyu_dataset():

    dataset_str = 'yiyu'
    data_path = '/mnt/mnt1/mzy/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年数据汇总_等级划分_最终.xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['抑郁症状等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 预处理
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

            #('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    preprocessor.fit(data)
    X_data = preprocessor.transform(data)

    # 收集特征名（用于画图 x 轴）
    #cat_names = preprocessor.named_transformers_['cat'][1].get_feature_names_out(cat_features)
    cat_encoder = preprocessor.named_transformers_['cat'][1]
    cat_names = cat_encoder.get_feature_names_out(cat_features)

    feat_names = list(cat_names) + num_features

    # 5. 标签
    label_map = {'抑郁症状不明显': 0, '中度抑郁症状': 1, '抑郁症状较明显': 2}
    labels = torch.tensor(data['抑郁症状等级'].map(label_map).values, dtype=torch.long)

    # 6. KNN 构图
    knn = NearestNeighbors(n_neighbors=5).fit(X_data)
    _, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = adj[j, i] = 1

    adj = adj.tocoo()                               # ★ 关键：转成 COO 才有 row/col

    edge_index = torch.tensor(
        np.vstack((adj.row, adj.col)), dtype=torch.long
    )

    # 7. 组装数据集（示例：SimpleNamespace；你也可以替换为自定义 NCDataset）
    dataset = SimpleNamespace()
    dataset.name = dataset_str
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels
    dataset.feat_names = feat_names                 # ★ 供后续画图使用

    # 8. 默认划分
    # 8. 默认划分 —— 先得到三组索引
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
    )

    # 转为张量
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx,  dtype=torch.long)

    # 保存到 dataset
    dataset.train_idx = train_idx
    dataset.valid_idx = valid_idx
    dataset.test_idx  = test_idx

    split_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # ---- 关键修改：让 get_idx_split 能接收任意参数 ----
    dataset.get_idx_split = lambda *args, **kwargs: split_dict
    # ------------------------------------------------

    return dataset

def load_jiaolv_dataset_missing():
    dataset_str = 'jiaolv'
    data_path = '/mnt/mnt1/mzy/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年体质+问卷数据汇总处理后_等级划分(带缺失).xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['焦虑等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 预处理
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

            #('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    preprocessor.fit(data)
    X_data = preprocessor.transform(data)

    # 收集特征名（用于画图 x 轴）
    #cat_names = preprocessor.named_transformers_['cat'][1].get_feature_names_out(cat_features)
    cat_encoder = preprocessor.named_transformers_['cat'][1]
    cat_names = cat_encoder.get_feature_names_out(cat_features)

    feat_names = list(cat_names) + num_features

    # 5. 标签
    label_map = {'焦虑症状不明显': 0, '中度焦虑症状': 1, '焦虑症状较明显': 2}
    labels = torch.tensor(data['焦虑等级'].map(label_map).values, dtype=torch.long)

    # 6. KNN 构图
    knn = NearestNeighbors(n_neighbors=5).fit(X_data)
    _, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = adj[j, i] = 1

    adj = adj.tocoo()                               # ★ 关键：转成 COO 才有 row/col

    edge_index = torch.tensor(
        np.vstack((adj.row, adj.col)), dtype=torch.long
    )

    # 7. 组装数据集（示例：SimpleNamespace；你也可以替换为自定义 NCDataset）
    dataset = SimpleNamespace()
    dataset.name = dataset_str
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels
    dataset.feat_names = feat_names                 # ★ 供后续画图使用

    # 8. 默认划分
    # 8. 默认划分 —— 先得到三组索引
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
    )

    # 转为张量
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx,  dtype=torch.long)

    # 保存到 dataset
    dataset.train_idx = train_idx
    dataset.valid_idx = valid_idx
    dataset.test_idx  = test_idx

    split_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # ---- 关键修改：让 get_idx_split 能接收任意参数 ----
    dataset.get_idx_split = lambda *args, **kwargs: split_dict
    # ------------------------------------------------

    return dataset


def load_yiyu_dataset_missing():

    dataset_str = 'yiyu'
    data_path = '/mnt/mnt1/mzy/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年体质+问卷数据汇总处理后_等级划分(带缺失).xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['抑郁等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 预处理
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

            #('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    preprocessor.fit(data)
    X_data = preprocessor.transform(data)

    # 收集特征名（用于画图 x 轴）
    #cat_names = preprocessor.named_transformers_['cat'][1].get_feature_names_out(cat_features)
    cat_encoder = preprocessor.named_transformers_['cat'][1]
    cat_names = cat_encoder.get_feature_names_out(cat_features)

    feat_names = list(cat_names) + num_features

    # 5. 标签
    label_map = {'抑郁症状不明显': 0, '中度抑郁症状': 1, '抑郁症状较明显': 2}
    labels = torch.tensor(data['抑郁等级'].map(label_map).values, dtype=torch.long)

    # 6. KNN 构图
    knn = NearestNeighbors(n_neighbors=5).fit(X_data)
    _, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = adj[j, i] = 1

    adj = adj.tocoo()                               # ★ 关键：转成 COO 才有 row/col

    edge_index = torch.tensor(
        np.vstack((adj.row, adj.col)), dtype=torch.long
    )

    # 7. 组装数据集（示例：SimpleNamespace；你也可以替换为自定义 NCDataset）
    dataset = SimpleNamespace()
    dataset.name = dataset_str
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels
    dataset.feat_names = feat_names                 # ★ 供后续画图使用

    # 8. 默认划分
    # 8. 默认划分 —— 先得到三组索引
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
    )

    # 转为张量
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx,  dtype=torch.long)

    # 保存到 dataset
    dataset.train_idx = train_idx
    dataset.valid_idx = valid_idx
    dataset.test_idx  = test_idx

    split_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # ---- 关键修改：让 get_idx_split 能接收任意参数 ----
    dataset.get_idx_split = lambda *args, **kwargs: split_dict
    # ------------------------------------------------

    return dataset

def load_PROTEINS_data():
    dataset_str = 'PROTEINS'
    data_path = '/mnt/mnt1/mzy/data/zijian/PROTEINS'

    # 加载 PROTEINS 数据（自动下载至指定路径）
    pyg_data = TUDataset(root=data_path, name=dataset_str)

    # 合并所有图为一个大图（node classification 任务）
    all_graphs = []
    all_labels = []
    offset = 0
    edge_list = []
    for data in pyg_data:
        G = to_networkx(data, to_undirected=True)
        n_nodes = G.number_of_nodes()
        mapping = {i: i + offset for i in range(n_nodes)}
        G = nx.relabel_nodes(G, mapping)
        edge_list += list(G.edges())
        all_graphs.append(G)
        all_labels.append(int(data.y.item()))
        offset += n_nodes

    num_nodes = offset
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 简化节点特征：使用单位特征（可以根据需要替换为实际 data.x 拼接）
    features = sp.identity(num_nodes, format='csr', dtype=np.float32)

    # 简化标签处理：将每个图的 label 复制给它的所有节点
    labels = np.zeros(num_nodes, dtype=np.int64)
    offset = 0
    for i, G in enumerate(all_graphs):
        n = G.number_of_nodes()
        labels[offset:offset + n] = all_labels[i]
        offset += n

    # 封装为 NCDataset 格式
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    # 默认划分
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx  = split['test']

    return dataset






def build_knn_graph(features, k=5):
    adj = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    adj = adj + adj.T  # 对称化
    adj.setdiag(0)
    edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)
    return edge_index


DATA_ROOT = "/mnt/mnt1/mzy/data/zijian"


def load_credit_fraud_data():
    dataset_str = 'CreditFraud'
    local_path = os.path.join(DATA_ROOT, 'creditcard.csv')
    if not os.path.exists(local_path):
        url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
        df = pd.read_csv(url)
        df.to_csv(local_path, index=False)
    else:
        df = pd.read_csv(local_path)

    features = df.drop(columns=['Class']).values
    labels = df['Class'].values
    features = StandardScaler().fit_transform(features)

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_heart_disease_data():
    dataset_str = 'HeartDisease'
    local_path = os.path.join(DATA_ROOT, 'heart.csv')
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

    if not os.path.exists(local_path):
        df = pd.read_csv(url, names=columns)
        df.to_csv(local_path, index=False)
    else:
        df = pd.read_csv(local_path)

    df.replace('?', np.nan, inplace=True)
    df = df.dropna().astype(float)
    df['target'] = (df['target'] > 0).astype(int)

    features = StandardScaler().fit_transform(df.drop(columns=['target']))
    labels = df['target'].values

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_nsl_kdd_data():
    dataset_str = 'NSL_KDD'
    local_path = os.path.join(DATA_ROOT, 'KDDTrain+.txt')
    url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt'

    if not os.path.exists(local_path):
        df = pd.read_csv(url, header=None)
        df.to_csv(local_path, index=False, header=False)
    else:
        df = pd.read_csv(local_path, header=None)

    labels = df.iloc[:, -2]
    features = df.iloc[:, :-2]

    for col in [1, 2, 3]:
        features[col] = LabelEncoder().fit_transform(features[col])

    labels = (labels != 'normal').astype(int)
    features = StandardScaler().fit_transform(features)

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_german_credit_data():
    dataset_str = 'GermanCredit'
    local_path = os.path.join(DATA_ROOT, 'german.data')
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

    if not os.path.exists(local_path):
        df = pd.read_csv(url, delimiter=' ', header=None)
        df.to_csv(local_path, index=False, header=False, sep=' ')
    else:
        df = pd.read_csv(local_path, delimiter=' ', header=None)

    labels = (df.iloc[:, -1].values == 2).astype(int)
    df = df.iloc[:, :-1]
    df = pd.get_dummies(df)
    features = StandardScaler().fit_transform(df.values)

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_traffic_metr_data():
    dataset_str = 'METR_LA'
    local_path = os.path.join(DATA_ROOT, 'metr_dummy.csv')

    # 模拟数据保存（真实 METR-LA 复杂，简化为分类任务）
    if not os.path.exists(local_path):
        dummy = pd.DataFrame(np.random.rand(207, 10))
        dummy.to_csv(local_path, index=False)
    else:
        dummy = pd.read_csv(local_path)

    features = dummy.values
    labels = (features[:, -1] > 0.5).astype(int)

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_gene_disease_association_prediction_with_gat():
    dataset_str = 'Gene_Disease_Association_Prediction_with_GAT'
    save_dir = '/mnt/mnt1/mzy/data/zijian'
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'GDA_associations.csv')

    # ---------- 1. 下载数据 ----------
    if not os.path.exists(csv_path):
        print('未找到 GDA_associations.csv，正在下载 …')
        url = ('https://raw.githubusercontent.com/'
               'RudraxDave/Gene_Disease_Association_Prediction_with_GAT/main/'
               'GDA_associations.csv')
        urlretrieve(url, csv_path)
        print('✅ 数据下载完成')

    # ---------- 2. 读取 + 清洗 ----------
    df = pd.read_csv(csv_path).dropna(subset=['score'])
    gene_cols     = ['gene_dsi', 'gene_dpi', 'gene_pli', 'protein_class']
    disease_cols  = ['disease_type', 'disease_semantic_type']
    disease_class = [
        'C01','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13',
        'C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24',
        'C25','C26','F01','F02','F03'
    ]

    # 2-1 基因节点特征
    gene_df = df[['geneid'] + gene_cols].drop_duplicates()
    gene_df['protein_class'] = pd.factorize(gene_df['protein_class'])[0]
    gene_df[gene_cols] = gene_df[gene_cols].apply(pd.to_numeric, errors='coerce')
    gene_df = gene_df.dropna(subset=gene_cols)

    # 2-2 疾病节点特征
    dis_df = df[['diseaseid', 'disease_class'] + disease_cols].drop_duplicates()
    for c in disease_cols:
        dis_df[c] = pd.factorize(dis_df[c])[0]
    for c in disease_class:
        dis_df[c] = 0
    for i, row in dis_df.iterrows():
        for c in str(row['disease_class']).split(';'):
            if c in disease_class:
                dis_df.at[i, c] = 1
    dis_df = dis_df.drop(columns=['disease_class'])

    # ---------- 3. 建立节点索引 ----------
    genes      = gene_df['geneid'].unique()
    diseases   = dis_df['diseaseid'].unique()
    gene2idx   = {g:i for i, g in enumerate(genes)}
    dis2idx    = {d:i+len(genes) for i, d in enumerate(diseases)}
    num_nodes  = len(genes) + len(diseases)

    # ---------- 4. 生成节点特征矩阵 ----------
    gene_feat = gene_df.set_index('geneid')[gene_cols].loc[genes].values
    dis_feat  = dis_df.set_index('diseaseid')[disease_cols + disease_class].loc[diseases].values
    total_dim = gene_feat.shape[1] + dis_feat.shape[1]
    # 对齐维度
    gene_pad  = np.zeros((len(genes), dis_feat.shape[1]))
    dis_pad   = np.zeros((len(diseases), gene_feat.shape[1]))
    X = np.vstack([np.hstack([gene_feat, gene_pad]),
                   np.hstack([dis_pad,  dis_feat])])
    X = StandardScaler().fit_transform(X)

    # ---------- 5. 正/负边 ----------
    pos_df = df[['geneid', 'diseaseid']].drop_duplicates()
    pos_src = pos_df['geneid'].map(gene2idx).to_numpy()
    pos_dst = pos_df['diseaseid'].map(dis2idx).to_numpy()

    # 负边随机采样（与正边同数量）
    neg_src, neg_dst = [], []
    gene_arr, dis_arr = list(genes), list(diseases)
    pos_set = set(zip(pos_src, pos_dst))
    while len(neg_src) < len(pos_src):
        g = gene2idx[random.choice(gene_arr)]
        d = dis2idx[random.choice(dis_arr)]
        if (g, d) not in pos_set:
            neg_src.append(g)
            neg_dst.append(d)

    # 无向图：边双向添加
    src = np.concatenate([pos_src, neg_src, pos_dst, neg_dst])
    dst = np.concatenate([pos_dst, neg_dst, pos_src, neg_src])
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    # edge_label 仅对“原向”边赋值
    edge_label_index = torch.tensor(
        np.vstack([np.concatenate([pos_src, neg_src]),
                   np.concatenate([pos_dst, neg_dst])]),
        dtype=torch.long
    )
    edge_label = torch.tensor(
        np.concatenate([np.ones(len(pos_src)), np.zeros(len(neg_src))]),
        dtype=torch.float
    )

    # ---------- 6. 打包 NCDataset ----------
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.edge_label_index = edge_label_index   # 👉 训练时用
    dataset.edge_label       = edge_label
    # 若框架仍要求 dataset.label，可随意占位
    dataset.label = edge_label

    # 划分索引（可以用框架自带，也可自行分）
    split = dataset.get_idx_split()     # 若此方法只管节点，可忽略
    dataset.train_idx = torch.arange(edge_label.size(0))   # 让你自己再切分
    dataset.valid_idx = torch.tensor([], dtype=torch.long)
    dataset.test_idx  = torch.tensor([], dtype=torch.long)

    return dataset






def load_ccle_data(seed=42):


    # 路径设置
    root = "/mnt/mnt1/mzy/data/zijian/ccle"
    expr_path = os.path.join(root, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    info_path = os.path.join(root, "Model.csv")

    # 加载数据
    df_expr = pd.read_csv(expr_path, index_col=0)
    df_info = pd.read_csv(info_path)

    print("表达矩阵 shape:", df_expr.shape)
    print("样本信息 shape:", df_info.shape)
    print("表达矩阵索引（前5行）:", df_expr.index[:5])
    print("样本信息列名：", df_info.columns.tolist())

    # 设置索引名为 ModelID，准备按 index 合并
    df_expr.index.name = "ModelID"

    # merge on ModelID（正确字段）
    df = df_info.merge(df_expr, on="ModelID")

    # 过滤无标签数据
    df = df[~df["OncotreePrimaryDisease"].isna()]
    class_counts = df["OncotreePrimaryDisease"].value_counts()
    df = df[df["OncotreePrimaryDisease"].map(class_counts) >= 10].copy()
    print("合并后样本数:", len(df))
    print("标签列频数:\n", df["OncotreePrimaryDisease"].value_counts())

    # 提取表达特征列（即 df_expr 的列）
    expr_cols = df_expr.columns
    X = df[expr_cols].fillna(0.0).astype(np.float32).values

    # 标签编码
    labels, _ = pd.factorize(df["OncotreePrimaryDisease"])

    # Keep the same stratified split for both model variants. Fit preprocessing
    # on training cells only so validation/test statistics cannot leak in.
    train_idx, other_idx = train_test_split(
        np.arange(len(labels)), train_size=0.7, random_state=seed, stratify=labels
    )
    valid_idx, test_idx = train_test_split(
        other_idx, train_size=0.5, random_state=seed, stratify=labels[other_idx]
    )
    X = StandardScaler().fit(X[train_idx]).transform(X).astype(np.float32)

    # 构建 k-NN 图
    adj = kneighbors_graph(X, n_neighbors=5, include_self=False)
    edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)

    # 构造 NCDataset 对象
    dataset = NCDataset("CCLE")
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X, dtype=torch.float32),
        'num_nodes': X.shape[0]
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = {
        "train": torch.tensor(train_idx, dtype=torch.long),
        "valid": torch.tensor(valid_idx, dtype=torch.long),
        "test": torch.tensor(test_idx, dtype=torch.long),
    }
    dataset.split_idx = split
    dataset.train_idx = split["train"]
    dataset.valid_idx = split["valid"]
    dataset.test_idx  = split["test"]

    return dataset







def load_breast_cancer_data():
    dataset_str = 'BreastCancer'
    save_path = '/mnt/mnt1/mzy/data/zijian/breast_cancer.csv'

    # 指定真实数据 URL
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    column_names = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_Cell_Size',
                    'Uniformity_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size',
                    'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

    if not os.path.exists(save_path):
        df = pd.read_csv(url, header=None, names=column_names)
        df.to_csv(save_path, index=False)
    else:
        df = pd.read_csv(save_path)

    df = df.drop(columns=['Sample_code_number'])
    df = df.replace('?', np.nan).dropna()
    df = df.astype(float)

    features = df.drop(columns=['Class']).values
    labels = df['Class'].values
    features = StandardScaler().fit_transform(features)

    edge_index = kneighbors_graph(features, n_neighbors=5, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.vstack(edge_index.nonzero()), dtype=torch.long)

    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset
import os, shutil, pickle
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Coauthor

# from your_lib import NCDataset  # 保持你的 NCDataset 导入

DATA_DIR = "/mnt/mnt1/mzy/data/zijian/coauthor/"  # 注意：这里以 coauthor 为结尾

def _load_splits(npz_path: str, num_nodes: int):
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    def _get_first(*cands):
        for k in cands:
            if k in keys:
                return z[k]
        return None

    # 优先掩码
    tm = _get_first("train_mask")
    vm = _get_first("val_mask", "valid_mask")
    sm = _get_first("test_mask")
    if tm is not None and vm is not None and sm is not None:
        tm = torch.as_tensor(tm).view(-1).bool()
        vm = torch.as_tensor(vm).view(-1).bool()
        sm = torch.as_tensor(sm).view(-1).bool()
        return tm, vm, sm, torch.where(tm)[0], torch.where(vm)[0], torch.where(sm)[0]

    # 其次索引
    ti = _get_first("train", "train_idx", "idx_train", "train_indices")
    vi = _get_first("val", "valid", "val_idx", "idx_val", "validation")
    si = _get_first("test", "test_idx", "idx_test")
    if ti is None or vi is None or si is None:
        raise KeyError(f"{npz_path} 中未找到可识别的划分键名。")

    ti = torch.as_tensor(ti, dtype=torch.long).view(-1)
    vi = torch.as_tensor(vi, dtype=torch.long).view(-1)
    si = torch.as_tensor(si, dtype=torch.long).view(-1)

    def to_mask(idx):
        m = torch.zeros(num_nodes, dtype=torch.bool)
        m[idx] = True
        return m

    return to_mask(ti), to_mask(vi), to_mask(si), ti, vi, si


def _safe_load_coauthor(name: str):
    """
    可靠地加载 PyG Coauthor 数据（root=DATA_DIR）。
    如遇到损坏的 raw/processed 缓存，会清理后重试一次。
    """
    transform = T.NormalizeFeatures()
    try:
        return Coauthor(root=DATA_DIR, name=name, transform=transform)
    except (pickle.UnpicklingError, EOFError, ValueError) as e:
        # 清理损坏缓存并重试
        base = os.path.join(DATA_DIR, name)
        for sub in ["raw", "processed"]:
            shutil.rmtree(os.path.join(base, sub), ignore_errors=True)
        return Coauthor(root=DATA_DIR, name=name, transform=transform)


def _build_nc_dataset(name: str, torch_data, split_npz_path: str):
    edge_index = torch_data.edge_index
    node_feat = torch_data.x
    label = torch_data.y
    num_nodes = int(torch_data.num_nodes)

    tr_mask, va_mask, te_mask, tr_idx, va_idx, te_idx = _load_splits(split_npz_path, num_nodes)

    dataset = NCDataset(name)
    dataset.graph = {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_feat": None,
        "num_nodes": num_nodes,
    }
    dataset.label = label
    dataset.train_mask, dataset.val_mask, dataset.test_mask = tr_mask, va_mask, te_mask
    dataset.train_idx,  dataset.val_idx,  dataset.test_idx  = tr_idx,  va_idx,  te_idx
    try:
        dataset.num_classes = int(label.max().item() + 1)
    except Exception:
        pass
    return dataset


def load_coauthor_CS_dataset():
    pyg_ds = _safe_load_coauthor("CS")  # ✅ root=DATA_DIR，由 PyG 自动拼 /CS/...
    data = pyg_ds[0]
    split_path = os.path.join(DATA_DIR, "coauthor-cs_split.npz")  # 你的 split 路径
    return _build_nc_dataset("coauthor-cs", data, split_path)



def load_amazon_computers_dataset():
    data_path = '/mnt/mnt1/mzy/data/zijian'
    dataset_str = 'amazon-computers'
    transform = T.NormalizeFeatures()

    torch_dataset = Amazon(
        root=f'{data_path}/amazon',
        name='Computers',
        transform=transform
    )
    data = torch_dataset[0]

    edge_index = data.edge_index
    features = data.x
    labels = data.y
    num_nodes = data.num_nodes

    # === 6. 构建 NCDataset 对象 ===
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': features.float(),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # === 7. 默认划分 train / valid / test ===
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset


def load_amazon_photo_dataset():
    data_path = '/mnt/mnt1/mzy/data/zijian'
    dataset_str = 'amazon-photo'
    transform = T.NormalizeFeatures()

    torch_dataset = Amazon(
        root=f'{data_path}/amazon',
        name='Photo',
        transform=transform
    )
    data = torch_dataset[0]

    edge_index = data.edge_index
    features = data.x
    labels = data.y
    num_nodes = data.num_nodes

    # === 6. 构建 NCDataset 对象 ===
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': features.float(),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # === 7. 默认划分 train / valid / test ===
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset


def load_coauthor_cs_dataset():
    data_path = '/mnt/mnt1/mzy/data/'
    dataset_str = 'coauthor-cs'
    transform = T.NormalizeFeatures()

    torch_dataset = Coauthor(
        root=f'{data_path}/Coauthor',
        name='CS',
        transform=transform
    )
    data = torch_dataset[0]

    edge_index = data.edge_index
    features = data.x
    labels = data.y
    num_nodes = data.num_nodes

    # === 6. 构建 NCDataset 对象 ===
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': features.float(),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # === 7. 默认划分 train / valid / test ===
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset



def load_coauthor_physics_dataset():
    data_path = '/mnt/mnt1/mzy/data/'
    dataset_str = 'coauthor-physics'
    transform = T.NormalizeFeatures()

    torch_dataset = Coauthor(
        root=f'{data_path}/Coauthor',
        name='Physics',
        transform=transform
    )
    data = torch_dataset[0]

    edge_index = data.edge_index
    features = data.x
    labels = data.y
    num_nodes = data.num_nodes

    # === 6. 构建 NCDataset 对象 ===
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': features.float(),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # === 7. 默认划分 train / valid / test ===
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset

def load_prepared_bio_dataset(name, data_dir):
    """Load a reproducible, single-label node classification graph prepared from public data."""
    file_path = os.path.join(data_dir, 'biomed', f'{name}.npz')
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f'{file_path} is missing; run prepare_bio_datasets.py --dataset {name} first'
        )
    with np.load(file_path, allow_pickle=False) as data:
        features = torch.from_numpy(data['features'].copy()).float()
        labels = torch.from_numpy(data['labels'].copy()).long()
        edges = torch.from_numpy(data['edge_index'].copy()).long()
        split = {
            key: torch.from_numpy(data[f'{key}_idx'].copy()).long()
            for key in ('train', 'valid', 'test')
        }
        class_names = data['class_names'].astype(str).tolist()
        fine_labels = (torch.from_numpy(np.unique(
            data['celltype_minor'].astype(str), return_inverse=True)[1].copy()).long()
            if 'celltype_minor' in data.files else None)
        patient_codes = (torch.from_numpy(np.unique(
            data['patient_ids'].astype(str), return_inverse=True)[1].copy()).long()
            if 'patient_ids' in data.files else None)
        celltype_minor = (data['celltype_minor'].astype(str).copy()
                          if 'celltype_minor' in data.files else None)
        patient_ids = (data['patient_ids'].astype(str).copy()
                       if 'patient_ids' in data.files else None)

    if labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError(f'Invalid node features or labels in {file_path}')
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f'Invalid edge_index in {file_path}')
    if min(len(split[key]) for key in split) == 0:
        raise ValueError(f'Empty train, validation, or test split in {file_path}')

    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': edges,
        'edge_feat': None,
        'node_feat': features,
        'num_nodes': int(features.shape[0]),
    }
    dataset.label = labels
    dataset.class_names = class_names
    dataset.cl_fine_labels = fine_labels
    dataset.patient_codes = patient_codes
    dataset.celltype_minor = celltype_minor
    dataset.patient_ids = patient_ids
    dataset.split_idx = split
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


if __name__ == '__main__':
    # load_airport()
    # load_wikipedia('squirrel')
    # load_wiki_new('chameleon')
    pass

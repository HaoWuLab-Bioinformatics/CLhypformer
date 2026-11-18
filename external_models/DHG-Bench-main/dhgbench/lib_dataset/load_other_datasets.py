#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pickle

import os.path as osp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_sparse import coalesce
from sklearn.feature_extraction.text import CountVectorizer

def load_dataset(path='../hyperGCN/data/', dataset = 'cora', train_percent = 0.025):

    # first load node features:
    with open(osp.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        if dataset in ['actor','amazon','pokec','twitch','twitch_ptbr','twitch_RU','twitch_ES']:
            pass
        else:
            features = features.todense()

    # then load node labels:
    with open(osp.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
            None,
            total_num_node_id_he_id,
            total_num_node_id_he_id)

    n_x = num_nodes
    data.n_x = n_x

    data.train_percent = train_percent
    data.num_hyperedges = len(hypergraph)

    return data

def load_large_dataset(path='../hyperGCN/data/', dataset = 'trivago', train_percent = 0.025):
    
    features = torch.load(osp.join(path,dataset,f'{dataset}_X_vec.pt'))

    # then load node labels:
    labels = torch.load(osp.join(path,dataset,f'{dataset}_Y.pt'))

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    hypergraph = torch.load(osp.join(path,dataset,f'{dataset}_E.pt'))

    num_hyperedges = hypergraph[1].max()+1
    print(f'number of hyperedges: {hypergraph[1].max()+1}')

    edge_idx = num_nodes
    hypergraph[1]+=edge_idx 
    node_list,edge_list = hypergraph[0].tolist(),hypergraph[1].tolist()

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = int)
    
    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    n_x = num_nodes
    data.n_x = n_x

    data.train_percent = train_percent
    data.num_hyperedges = num_hyperedges
    
    return data

def load_yelp_dataset(path='../data/raw_data/yelp_raw_datasets/', dataset = 'yelp', 
        name_dictionary_size = 1000,
        train_percent = 0.025):

    latlong = pd.read_csv(osp.join(path, 'yelp_restaurant_latlong.csv')).values

    loc = pd.read_csv(osp.join(path, 'yelp_restaurant_locations.csv'))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    vectorizer = CountVectorizer(max_features = name_dictionary_size, stop_words = 'english', strip_accents = 'ascii')
    res_name = pd.read_csv(osp.join(path, 'yelp_restaurant_name.csv')).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()

    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    df_labels = pd.read_csv(osp.join(path, 'yelp_restaurant_business_stars.csv'))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    H = pd.read_csv(osp.join(path, 'yelp_restaurant_incidence_H.csv'))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes

    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    n_x = num_nodes
    data.n_x = n_x
    
    data.train_percent = train_percent
    data.num_hyperedges = H.he.values.max()
    
    return data

def load_cornell_dataset(path='../data/raw_data/', dataset = 'amazon', 
        feature_noise = 0.1,
        feature_dim = None,
        train_percent = 0.025):

    # first load node labels
    df_labels = pd.read_csv(osp.join(path, dataset, f'node-labels-{dataset}.txt'), names = ['node_label'])
    num_nodes = df_labels.shape[0]
    labels = df_labels.values.flatten()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_nodes, num_classes))

    features[np.arange(num_nodes), labels - 1] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype = features.dtype)
        features = np.hstack((features, zero_col))

    features = np.random.normal(features, feature_noise, features.shape)
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    p2hyperedge_list = osp.join(path, dataset, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list, 
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    n_x = num_nodes
    data.n_x = n_x
    
    data.train_percent = train_percent
    data.num_hyperedges = he_id - num_nodes
    
    return data

def load_LE_dataset(path=None, dataset="ModelNet40", train_percent = 0.025):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))
    
    file_name = f'{dataset}.content'
    p2idx_features_labels = osp.join(path, dataset, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    print ('load features')
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    file_name = f'{dataset}.edges'
    p2edges_unordered = osp.join(path, dataset, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print ('load edges')

    # From adjacency matrix to edge_list
    edge_index = edges.T 
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1
    
    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1
    
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    
    # build torch data class
    data = Data(
            x = torch.FloatTensor(np.array(features[:num_nodes].todense())), 
            edge_index = torch.LongTensor(edge_index),
            y = labels[:num_nodes])

    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    n_x = num_nodes
    num_class = len(np.unique(labels[:num_nodes].numpy()))
    data.n_x = n_x
    data.y=data.y-data.y.min()
    
    data.train_percent = train_percent
    data.num_hyperedges = num_he
    
    return data

def load_disease_dataset(path='../data/data/hete_data/', dataset='disease_nc', train_percent=0.025):
    """
    专门用于加载 disease_nc 数据集
    文件格式: .edges.csv, .feats.npz, .labels.npy
    """
    print(f'Loading {dataset} dataset from {path}...')
    dataset_path = osp.join(path, dataset)
    
    # 1. 先加载节点标签 (.npy) 以确定节点数量
    label_file = osp.join(dataset_path, f'{dataset}.labels.npy')
    labels = np.load(label_file, allow_pickle=True)
    if labels.ndim > 1:
        labels = labels.flatten()
    num_nodes = len(labels)
    print(f'Number of nodes: {num_nodes}')
    
    # 2. 加载节点特征 (.npz)
    feat_file = osp.join(dataset_path, f'{dataset}.feats.npz')
    feat_data = np.load(feat_file, allow_pickle=True)
    features = feat_data[feat_data.files[0]] if hasattr(feat_data, 'files') else feat_data
    
    if sp.issparse(features):
        features = features.todense()
    if features.ndim == 1:
        features = features.reshape(num_nodes, -1)
    print(f'Feature shape: {features.shape}')
    
    # 3. 加载超边文件 (.csv)
    edge_file = osp.join(dataset_path, f'{dataset}.edges.csv')
    edge_df = pd.read_csv(edge_file)
    
    if edge_df.shape[1] == 2:
        node_list = edge_df.iloc[:, 0].values
        he_list = edge_df.iloc[:, 1].values
    else:
        raise ValueError(f'Unexpected edge file format: {edge_df.shape}')
    
    # 关键修复：将超边 ID 重映射为从 0 开始的连续整数，防止 coalesce 索引越界
    unique_he = np.unique(he_list)
    he_map = {old: new for new, old in enumerate(unique_he)}
    he_list = np.array([he_map[x] for x in he_list])
    num_hyperedges = len(unique_he)
    print(f'Number of hyperedges: {num_hyperedges} (remapped)')
    
    # 4. 构建 PyG Data 对象
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    edge_idx_offset = num_nodes
    he_shifted = he_list + edge_idx_offset
    
    node_list_tensor = torch.LongTensor(node_list)
    he_shifted_tensor = torch.LongTensor(he_shifted)
    
    edge_index = torch.stack([
        torch.cat([node_list_tensor, he_shifted_tensor]),
        torch.cat([he_shifted_tensor, node_list_tensor])
    ], dim=0)
    
    data = Data(x=features, edge_index=edge_index, y=labels)
    
    total_num = num_nodes + num_hyperedges
    # 现在索引最大值必然小于 total_num，coalesce 安全通过
    data.edge_index, data.edge_attr = coalesce(data.edge_index, None, total_num, total_num)
    
    data.n_x = num_nodes
    data.train_percent = train_percent
    # 关键修复：num_hyperedges 必须是整数，与标准数据集保持一致
    data.num_hyperedges = num_hyperedges
    
    # 验证 edge_index 格式是否正确
    expected_max = num_nodes + num_hyperedges - 1
    actual_max = data.edge_index[0].max().item()
    if actual_max != expected_max:
        print(f'[WARNING] edge_index max mismatch! Expected {expected_max}, got {actual_max}')
        print(f'This will cause ExtractV2E to fail silently!')
    
    print(f'Dataset loaded successfully! Nodes: {num_nodes}, Feat: {features.shape[1]}, HE: {num_hyperedges}')
    return data


def load_email_dataset(path='../data/data/hete_data/', dataset='email-Eu-core', train_percent=0.025):
    """
    修复后的 email-Eu-core 加载函数
    解决 20% 准确率问题的关键：修复特征构建与标签映射
    """
    print(f'Loading {dataset} dataset from {path}...')
    dataset_path = osp.join(path, dataset)

    # 1. 加载并修复节点标签
    label_file = osp.join(dataset_path, f'{dataset}-department-labels.txt')
    label_data = np.loadtxt(label_file, dtype=int)

    # 确保节点按 ID 排序
    if label_data.ndim > 1:
        label_data = label_data[label_data[:, 0].argsort()]
        labels = label_data[:, 1]
    else:
        labels = label_data

    num_nodes = len(labels)

    # 【关键】：重新映射标签为 0 到 C-1 的连续整数
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    num_classes = len(unique_labels)

    print(f'Nodes: {num_nodes}, Classes: {num_classes} (mapped to continuous)')

    # 2. 构建有效的节点特征 (取代原有的 One-hot Label)
    # 计算节点度数作为基础结构特征
    edge_file = osp.join(dataset_path, f'{dataset}.txt')
    node_degree = np.zeros(num_nodes)
    with open(edge_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                u, v = int(parts[0]), int(parts[1])
                if u < num_nodes: node_degree[u] += 1
                if v < num_nodes: node_degree[v] += 1

    # 归一化度数特征
    deg_feat = (node_degree - node_degree.mean()) / (node_degree.std() + 1e-8)
    deg_feat = deg_feat.reshape(-1, 1)

    # 补充随机初始化特征（提供区分性维度）
    np.random.seed(42)
    random_feat = np.random.randn(num_nodes, 32)
    features = np.hstack([deg_feat, random_feat])

    # 3. 构建超图结构 (Sender-centric)
    node_list, he_list = [], []
    edges = np.loadtxt(edge_file, dtype=int)
    edge_df = pd.DataFrame(edges, columns=['src', 'dst'])

    # 将同一个发件人的所有联系人视为一条超边
    grouped = edge_df.groupby('src')['dst'].apply(list).reset_index()
    he_id = 0
    for _, row in grouped.iterrows():
        src, dsts = row['src'], row['dst']
        nodes = list(set([src] + dsts))  # 包含发件人和收件人
        node_list.extend(nodes)
        he_list.extend([he_id] * len(nodes))
        he_id += 1
    num_hyperedges = he_id

    # 4. 封装 PyG 数据对象
    x = torch.FloatTensor(features)
    y = torch.LongTensor(labels)

    # 构建双分图：Node <-> Hyperedge
    # he_id 需要偏移 num_nodes 长度以区分节点
    edge_index = torch.stack([
        torch.cat([torch.LongTensor(node_list), torch.LongTensor(he_list) + num_nodes]),
        torch.cat([torch.LongTensor(he_list) + num_nodes, torch.LongTensor(node_list)])
    ], dim=0)

    data = Data(x=x, edge_index=edge_index, y=y)

    # 移除重复边并规范化
    total_ids = num_nodes + num_hyperedges
    data.edge_index, _ = coalesce(data.edge_index, None, total_ids, total_ids)

    data.n_x = num_nodes
    data.num_hyperedges = num_hyperedges
    data.train_percent = train_percent

    return data

def load_polblogs_dataset(path='../data/data/hete_data/', dataset='PolBlogs', train_percent=0.025):
    """
    加载 PolBlogs 政治博客数据集
    文件格式: adjacency.tsv (src\tdst\tTrue), labels.tsv, names.tsv
    使用博客名称的TF-IDF特征
    """
    print(f'Loading {dataset} dataset from {path}...')
    dataset_path = osp.join(path, dataset, 'raw')
    
    # 1. 加载标签
    label_file = osp.join(dataset_path, 'labels.tsv')
    labels = np.loadtxt(label_file, dtype=int, delimiter='\t')
    if labels.ndim > 1:
        labels = labels.flatten()
    num_nodes = len(labels)
    num_classes = len(np.unique(labels))
    print(f'Number of nodes: {num_nodes}, Classes: {num_classes}')
    print(f'Label distribution: {np.bincount(labels)}')
    
    # 2. 加载博客名称并提取TF-IDF特征
    names_file = osp.join(dataset_path, 'names.tsv')
    with open(names_file, 'r', encoding='utf-8') as f:
        blog_names = [line.strip() for line in f]
    
    # 使用TF-IDF提取文本特征
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=128, stop_words='english')
    features = tfidf.fit_transform(blog_names).toarray().astype(np.float32)
    
    print(f'TF-IDF feature shape: {features.shape}')
    
    # 3. 加载边文件
    edge_file = osp.join(dataset_path, 'adjacency.tsv')
    edge_list = []
    
    with open(edge_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                src, dst = int(parts[0]), int(parts[1])
                if src < num_nodes and dst < num_nodes:
                    edge_list.append((src, dst))
    
    print(f'Number of edges: {len(edge_list)}')
    
    # 4. 构建超图：按源节点聚合
    print("Converting graph edges to hypergraph (source-centric)...")
    edge_df = pd.DataFrame(edge_list, columns=['src', 'dst'])
    grouped = edge_df.groupby('src')['dst'].apply(list).reset_index()
    
    node_list = []
    he_list = []
    he_id = 0
    
    for _, row in grouped.iterrows():
        src = row['src']
        dsts = row['dst']
        nodes = list(set([src] + dsts))
        node_list.extend(nodes)
        he_list.extend([he_id] * len(nodes))
        he_id += 1
    
    num_hyperedges = he_id
    print(f'Number of hyperedges: {num_hyperedges}')
    
    # 5. 构建 PyG Data 对象
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    edge_idx_offset = num_nodes
    he_shifted = np.array(he_list) + edge_idx_offset
    
    node_list_tensor = torch.LongTensor(node_list)
    he_shifted_tensor = torch.LongTensor(he_shifted)
    
    edge_index = torch.stack([
        torch.cat([node_list_tensor, he_shifted_tensor]),
        torch.cat([he_shifted_tensor, node_list_tensor])
    ], dim=0)
    
    data = Data(x=features, edge_index=edge_index, y=labels)
    
    total_num = num_nodes + num_hyperedges
    data.edge_index, data.edge_attr = coalesce(data.edge_index, None, total_num, total_num)
    
    data.n_x = num_nodes
    data.train_percent = train_percent
    data.num_hyperedges = num_hyperedges
    
    print(f'Dataset loaded successfully! Nodes: {num_nodes}, Feat Dim: {features.shape[1]}, HE: {num_hyperedges}')
    return data

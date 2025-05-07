import os
import random
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.utils.rnn import pad_sequence
import time
import dgl

try:
    import ipdb
except:
    pass

def strfy_args(args):
    pass

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

def set_seeds(seed):
    """
    Sets the seed for various random number generators to ensure reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.seed(seed)

class Device(object):
    """
    A simple wrapper class for handling device (CPU/GPU) operations in PyTorch.
    Attributes:
        dv_name: The name of the device (e.g., 'cuda:0' for GPU, 'cpu' for CPU).
    """
    def __init__(self, device_name):
        """
        Initializes the Device class with a specified device name.
        """
        self.dv_name = device_name
    
    def transfer(self, x):
        """
        Transfers a given tensor to the specified device.
        This is useful in PyTorch when working with models and tensors to ensure
        that they are on the correct device (CPU or GPU) for computation
        Returns:
            The tensor transferred to the specified device.
        """
        x.to(self.dv_name)

def save_model(model, i, save_dir, optimizer=None, scheduler=None):
    """
    Saves the current model state along with its optimizer and scheduler states.
    The function checks if both an optimizer and a scheduler are provided.
    If they are, it saves the model state, optimizer state, and scheduler state together.
    If not, it only saves the model state. The saved file is named 'model_{i}.xhr',
    where {i} is replaced by the provided identifier.
    """
    if optimizer is not None and scheduler is not None:
        torch.save({
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict()
            }, os.path.join(save_dir, 'model_{}.xhr'.format(i)))
    else:
        torch.save({
            "state_dict":model.state_dict(),
            }, os.path.join(save_dir, 'model_{}.xhr'.format(i)))

def path_exist(path):
    """
    Checks if a directory exists, and if not, creates it.
    This function first checks if the directory specified by 'path' exists.
    If the directory does not exist, it creates the directory along with any
    necessary intermediate directories.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def filt_params(named_params, filt_key):
    """
    Filters parameters based on a specified key.
    Returns:
        list
    This function iterates through 'named_params', filtering out and returning
    only those parameters whose names contain the specified 'filt_key'.
    """
    filted = []
    for name, par in named_params:
        if filt_key in name:
            filted.append(par)
    return filted

def delete_models(epochs, base_path):
    """
    Deletes model files corresponding to specific training epochs.
    For each epoch number in 'epochs', this function constructs the file name of
    the corresponding saved model and deletes it from the file system.
    """
    for e in epochs:
        os.remove(os.path.join(base_path, "model_{}.xhr".format(e)))

class Logger(object):
    """
    Logger class for recording training processes and results.

    This class provides functionalities for logging messages both to the console
    and to a file, with time stamps included for each entry. It's useful for
    tracking the progress and results of machine learning experiments.

    Attributes:
        log_file: A file object for writing logs to a file.
        is_write_file (bool): Determines whether to write logs to a file.
    """

    def __init__(self, log_path, name, seed, is_write_file=True):
        cur_time = time.strftime("%m-%d-%H:%M", time.localtime())
        self.is_write_file = is_write_file
        if self.is_write_file:
            self.log_file = open(os.path.join(log_path, "%s %s(%d).log" % (cur_time, name, seed)), 'w')
    
    def log(self, log_str):
        """
        Logs a given string with a time stamp.
        """
        out_str = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {log_str}"
        print(out_str)
        if self.is_write_file:
            self.log_file.write(out_str+'\n')
            self.log_file.flush()
    
    def close_log(self):
        """
        Closes the log file if file logging is enabled.
        """
        if self.is_write_file:
            self.log_file.close()

def checkin_graph_struct(o_ck):
    """
    Constructs graph structures from check-in sequences.
    Returns:
        tuple
    This function processes check-in sequences to create graph structures for each sequence.
    It constructs adjacency matrices representing the connections between nodes in the sequences,
    normalizes these matrices, and prepares alias tensors for indexing.
    """
    inputs = o_ck.cpu().numpy()
    items, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    max_n_node = np.max(n_node)
    for u_input in inputs:
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0) # out degree of every u
        u_sum_in[np.where(u_sum_in == 0)] = 1 # avoid divided by zero
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) # re-index
    alias = torch.LongTensor(alias_inputs)
    A = torch.FloatTensor(A)
    items = torch.LongTensor(items)
    
    return alias, A, items

def cuda():
    """
    Checks if CUDA is available for PyTorch operations.
    Returns:
        bool
    This function is a utility to quickly check the availability of CUDA, which is used for GPU-based computations.
    """
    return torch.cuda.is_available()
    #return False

def eval_sampling(region_poi, batch_size, device, duplicate=False, k=10):
    """
    Samples Points of Interest (POIs) and their corresponding regions for evaluation.
    Returns:
        tuple
    If 'duplicate' is False, it samples 'k' distinct POIs for each region, for each batch.
    If 'duplicate' is True, it samples 'k' POIs once per region and duplicates this sample across the batch.
    """
    if not duplicate:
        poi_samples, region_samples = [], []
        for r, v in region_poi.items():
            s = np.random.choice(list(v), (batch_size, k))
            poi_samples.append(torch.from_numpy(s))
            region_samples.append(torch.from_numpy(np.full((batch_size, k), r)))
        poi_samples = torch.cat(poi_samples, dim=1).to(device)
        region_samples = torch.cat(region_samples, dim=1).to(device)
        return poi_samples, region_samples
    else:
        poi_samples = []
        region_samples = []
        for r, v in region_poi.items():
            s = np.random.choice(list(v), (1, k))
            poi_samples.append(torch.from_numpy(s))
            region_samples.append(torch.from_numpy(np.full((1, k), r)))
        poi_samples = torch.cat(poi_samples, dim=1).to(device)
        region_samples = torch.cat(region_samples, dim=1).to(device)
        return poi_samples, region_samples

def collate_fn(batch):
    """
    Custom collation function for batching data in a DataLoader.
    This function is used to process a batch of data items and ensure they are in a consistent format,
    suitable for model training or evaluation.
    Returns:
        tuple
    """
    uid, ori_ck, dst_ck, masked_dst_ck, o_hour, d_hour, masked_d_h, ori_t, dst_t, ori_l, dst_l, ori_rg, dst_rg = zip(*batch)

    pad_ori_ck = pad_sequence(ori_ck, batch_first=True)
    pad_dst_ck = pad_sequence(dst_ck, batch_first=True)
    pad_masked_dst_ck = pad_sequence(masked_dst_ck, batch_first=True)
    pad_o_hour = pad_sequence(o_hour, batch_first=True)
    pad_d_hour = pad_sequence(d_hour, batch_first=True)
    pad_masked_d_hour = pad_sequence(masked_d_h, batch_first=True)
    pad_ori_t = pad_sequence(ori_t, batch_first=True)
    pad_ori_l = pad_sequence(ori_l, batch_first=True)
    pad_dst_t = pad_sequence(dst_t, batch_first=True)
    pad_dst_l = pad_sequence(dst_l, batch_first=True)
    ori_rg = torch.LongTensor(ori_rg)
    dst_rg = torch.LongTensor(dst_rg)
    uid = torch.LongTensor(uid)
    # 为 ori_ck 生成 pad mask：有效数据为 True，padding 为 False
    lens_ori = torch.tensor([len(seq) for seq in ori_ck], dtype=torch.long)
    max_len_ori = pad_ori_ck.size(1)
    ori_pad = torch.arange(max_len_ori).unsqueeze(0).expand(len(ori_ck), max_len_ori) < lens_ori.unsqueeze(1)
    # 为 AGG token 位置增加一列（设为 True）
    ori_pad = torch.cat([ori_pad, torch.ones(len(ori_ck), 1, dtype=torch.bool)], dim=1)

    # 同样，为 dst_ck 生成 pad mask
    lens_dst = torch.tensor([len(seq) for seq in dst_ck], dtype=torch.long)
    max_len_dst = pad_dst_ck.size(1)
    dst_pad = torch.arange(max_len_dst).unsqueeze(0).expand(len(dst_ck), max_len_dst) < lens_dst.unsqueeze(1)
    # # 如果你也想给 dst_ck 增加 AGG token 位置
    dst_pad = torch.cat([dst_pad, torch.ones(len(dst_ck), 1, dtype=torch.bool)], dim=1)

    return uid, pad_ori_ck, pad_dst_ck, pad_masked_dst_ck, pad_o_hour, pad_d_hour, pad_masked_d_hour, pad_ori_t, pad_dst_t, pad_ori_l, pad_dst_l, ori_pad, dst_pad, ori_rg, dst_rg


def normalize_gat(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Converts a scipy sparse matrix to a PyTorch sparse tensor.
    Returns:
        torch.sparse.FloatTensor
    This function transforms a scipy sparse matrix into a PyTorch sparse tensor.
    It is useful for using scipy-based sparse data structures in PyTorch models,
    particularly in graph-based neural networks.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    ret = torch.sparse.FloatTensor(indices, values, shape)
    if cuda(): ret = ret.cuda()
    return ret

def read_graph(dir_path):
    """
    Reads graph data from files and returns the adjacency matrices.
    Returns:
        tuple
    This function reads the adjacency matrices of the POI-POI graph and the Region-POI graph
    from the specified directory. These matrices are typically used in graph-based recommender
    systems or similar applications.
    """
    poi_poi_graph = sp.load_npz(os.path.join(dir_path, 'poi_poi_graph_ada.npz'))
    region_poi_graph = sp.load_npz(os.path.join(dir_path, 'region_poi_graph.npz'))
    return poi_poi_graph, region_poi_graph

def subgraph(adj, select_indice, n_poi, n_region):
    """
    Extracts a subgraph from the given adjacency matrix.
    Returns:
        torch.Tensor
    This function creates a subgraph by selecting rows and columns from the full
    adjacency matrix corresponding to the given indices. It is useful in graph-based models
    where a smaller portion of the graph is required for specific computations.
    """
    select_indice = select_indice.tolist()

    col_indice = [select_indice]
    row_indice = [[i] for i in select_indice]

    sub_adj = adj[row_indice, col_indice]
    sub_adj = sparse_mx_to_torch_sparse_tensor(sub_adj)
    if cuda(): sub_adj = sub_adj.cuda()

    return sub_adj.to_dense()

def convert_sp_mat_to_sp_tensor(X):
    """
    Converts a scipy sparse matrix to a PyTorch sparse tensor.
    Returns:
        torch.sparse.FloatTensor
    This function is useful for converting scipy-based sparse data structures into
    PyTorch sparse tensors, enabling their use in PyTorch models.
    """
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)


# generate transfer matrix
def poi_adjacent(train_data, poi_size):
    AM = np.zeros((poi_size, poi_size), dtype=int)
    traj_list = [train_data.dataset[i][2].tolist() for i in train_data.indices]
    for traj in traj_list:
        for index in range(len(traj)-1):
            curr_poi = traj[index]
            next_poi = traj[index + 1]
            AM[curr_poi][next_poi] += 1  # [v,v] pi -> pi+1

    row_sums = AM.sum(axis=1)
    AM = AM / row_sums[:, np.newaxis]
    AM[np.isnan(AM)] = 0

    return AM.astype(np.float32)

def poi_position(train_data, poi_size, max_length):
    PM = np.zeros((poi_size, max_length))
    traj_list = [train_data.dataset[i][2].tolist() for i in train_data.indices]
    for traj in traj_list:
        for index in range(len(traj)):
            PM[traj[index]][index] += 1  # [v,l_max]

    row_sums = PM.sum(axis=1)
    PM = PM / row_sums[:, np.newaxis]
    PM[np.isnan(PM)] = 0

    zero_counts = np.count_nonzero(PM == 0, axis=0)
    total_points = PM.shape[0]
    confidence = zero_counts / total_points
    confidence = [min(0.5, val) for val in confidence] # 如果一个位置上很多 POI 都没有出现（零比例高），说明该位置的信息较为稀疏，只有少数几个 POI 在该位置上频繁出现，可能代表该位置具有较强的区分性或“信心”
    # print(PM)
    return PM.astype(np.float32), confidence
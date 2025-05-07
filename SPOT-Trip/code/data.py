from random import shuffle, choice
import numpy as np
import scipy.sparse as sp
from copy import copy
from collections import defaultdict
from torch.utils.data import Dataset, Subset
import pandas as pd
import collections
from os.path import join
import torch
import json
import dgl
import os
import pickle
from collections import Counter
try:
    import ipdb
except:
    pass
import matplotlib.pyplot as plt
import pytz
from datetime import datetime, timezone
from utils import *

def convert_timestamp(region, timestamp, city_tz_mapping):

    ts_sec = int(timestamp)
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    local_tz = pytz.timezone(city_tz_mapping.get(region, "UTC"))
    local_dt = dt_utc.astimezone(local_tz)
    return local_dt, local_dt.hour


def compute_trajectory_duration(trajectory):
    times = []
    for point in trajectory:
        ts = point[2]
        ts_sec = int(ts)
        dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        times.append(dt)
    if not times:
        return None, None, None, None, None
    min_time = min(times)
    max_time = max(times)
    duration = max_time - min_time
    return duration, min_time, max_time

class KGDataset(Dataset):
    """
    A custom dataset class for handling knowledge graph (KG) data in machine learning models.
    This class processes and stores knowledge graph data, including entities, relations, and triples.
    """
    def __init__(self, args):
        kg_data = pd.read_csv(args.kg_path, sep='\t', names=['h', 'r', 't'], engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)
        self.args = args

    @property
    def entity_count(self):
        """
        Returns the total count of unique entities in the knowledge graph.
        Returns:
            int
        """
        # start from one
        return self.kg_data['t'].max() + 2

    @property
    def relation_count(self):
        """
        Returns the total count of unique relations in the knowledge graph.
        Returns:
            int
        """
        return self.kg_data['r'].max()+2

    def get_kg_dict(self, poi_num):
        """
        Generates a dictionary with POI-specific KG entity and relation information.
        Returns:
            dict
        """
        entity_num = self.args.entity_num_per_poi # 2
        p2es = dict()
        p2rs = dict()
        for poi in range(poi_num):
            rts = self.kg_dict.get(poi, False)
            if rts:
                tails = list(map(lambda x:x[1], rts))
                relations = list(map(lambda x:x[0], rts))
                if(len(tails) >= entity_num):
                    p2es[poi] = torch.IntTensor(tails).to(self.args.device)[:entity_num]
                    p2rs[poi] = torch.IntTensor(relations).to(self.args.device)[:entity_num]
                else:
                    # last embedding pos as padding idx
                    tails.extend([self.entity_count]*(entity_num-len(tails)))
                    relations.extend([self.relation_count]*(entity_num-len(relations)))
                    p2es[poi] = torch.IntTensor(tails).to(self.args.device)
                    p2rs[poi] = torch.IntTensor(relations).to(self.args.device)
            else:
                p2es[poi] = torch.IntTensor([self.entity_count]*entity_num).to(self.args.device)
                p2rs[poi] = torch.IntTensor([self.relation_count]*entity_num).to(self.args.device)
        return p2es, p2rs


    def generate_kg_data(self, kg_data):
        """
        Constructs a dictionary representation of the knowledge graph.
        Returns:
            dict
            list
        """
        kg_dict = collections.defaultdict(list)
        for row in kg_data.iterrows():
            h, r, t = row[1]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    def __len__(self):
        """
        Returns the total number of head entities in the knowledge graph.
        Returns:
            int
        """
        return len(self.kg_dict)

    def __getitem__(self, index):
        """
        Retrieves a KG triple (head, relation, positive tail, negative tail) at a specified index.
        Returns:
            tuple
        """
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail

class TravelDataset(Dataset):
    """
    A custom dataset class for handling travel-related data for machine learning models.
    This class processes and stores data related to points of interest (POIs), regions,
    user transactions, and associated features for use in models focusing on travel data.
    """
    def __init__(self, args, ori_data_path, dst_data_path, trans_data_path):
        ori_raw = list(map(lambda x: x.strip().split('\t'), open(ori_data_path, 'r')))
        dst_raw = list(map(lambda x: x.strip().split('\t'), open(dst_data_path, 'r')))
        trans_raw = list(map(lambda x: x.strip().split('\t'), open(trans_data_path, 'r')))
        self.args = args
        self.poi_idx = {}
        self.region_idx = {}
        self.tag_idx = {}
        self.region_poi = defaultdict(set)

        self.trans = []
        self.feats = []
        self.uids = []

        with open(f"../{self.args.dataset_name}/city_tz_mapping.pkl", "rb") as f:
            city_tz_mapping = pickle.load(f)

        for i in trans_raw:
            uid, cuid, ori_region, dst_region = i
            if ori_region not in self.region_idx: 
                self.region_idx[ori_region] = len(self.region_idx) 
            if dst_region not in self.region_idx:
                self.region_idx[dst_region] = len(self.region_idx)
            self.trans.append((self.region_idx[ori_region], self.region_idx[dst_region]))
            self.uids.append(int(uid))


        # for i in ori_raw + dst_raw:
        #     uid, cuid, _, bid, timestamp, std_tag = i
        #     if bid not in self.poi_idx:
        #         self.poi_idx[bid] = len(self.poi_idx) + 1
        #     if std_tag not in self.tag_idx:
        #         self.tag_idx[std_tag] = len(self.tag_idx)
        bid_counter = Counter([i[3] for i in (ori_raw + dst_raw)])
        tag_counter = Counter([i[5] for i in (ori_raw + dst_raw)])

        sorted_bids = [bid for bid, freq in bid_counter.most_common()]
        sorted_tags = [tag for tag, count in tag_counter.most_common()]

        self.poi_idx = {bid: idx for idx, bid in enumerate(sorted_bids, start=1)}
        self.tag_idx = {tag: idx for idx, tag in enumerate(sorted_tags, start=1)}

        with open(f"../{self.args.dataset_name}/poi_coord.pkl", 'rb') as f:
            poi_coord = pickle.load(f)

        all_lats = [coord[0] for coord in poi_coord.values()]
        all_lons = [coord[1] for coord in poi_coord.values()]
        global_lat_min = min(all_lats)
        global_lat_max = max(all_lats)
        global_lon_min = min(all_lons)
        global_lon_max = max(all_lons)
        lat_range = global_lat_max - global_lat_min if global_lat_max != global_lat_min else 1
        lon_range = global_lon_max - global_lon_min if global_lon_max != global_lon_min else 1

        def normalize_coord(coord):
            lat, lon = coord
            return np.array(((lat - global_lat_min) / lat_range, (lon - global_lon_min) / lon_range))

        self.poi_coord_norm = {poi: normalize_coord(coord) for poi, coord in poi_coord.items()}

        self.oris = []
        self.dsts = []

        ori_buffer = []
        train_buffer = []
        dst_buffer = []

        last_uid = '0'
        for i in ori_raw:
            uid, cuid, rid, bid, timestamp, std_tag = i
            self.region_poi[self.trans[int(uid)][0]].add(self.poi_idx[bid])
            local_time, local_hour = convert_timestamp(rid, timestamp, city_tz_mapping)
            if uid != last_uid:
                self.oris.append(ori_buffer)
                ori_buffer = []
                train_buffer = []
                last_uid = uid
            ori_buffer.append((self.poi_idx[bid], self.tag_idx[std_tag], float(timestamp), local_hour))
            train_buffer.append(self.poi_idx[bid])
        self.oris.append(ori_buffer)

        last_uid = '0'
        for i in dst_raw:
            uid, cuid, rid, bid, timestamp, std_tag = i
            self.region_poi[self.trans[int(uid)][1]].add(self.poi_idx[bid])
            local_time, local_hour = convert_timestamp(rid, timestamp, city_tz_mapping)
            if uid != last_uid:
                self.dsts.append(dst_buffer)
                dst_buffer = []
                last_uid = uid
            dst_buffer.append((self.poi_idx[bid], self.tag_idx[std_tag], float(timestamp), local_hour))
        self.dsts.append(dst_buffer)
        self.oris_norm = []
        self.oris_duration = []
        for traj in self.oris:
            timestamps = np.array([item[2] for item in traj])
            t_min, t_max = timestamps.min(), timestamps.max()
            duration = t_max - t_min
            t_range = duration if duration != 0 else 1
            norm_traj = []
            norm_times = (timestamps - t_min) / t_range
            for idx, (poi, tag, ts, local_hour) in enumerate(traj):
                norm_time = norm_times[idx]
                norm_traj.append((poi, tag, ts, local_hour, norm_time, self.poi_coord_norm[poi]))
            self.oris_norm.append(norm_traj)
            self.oris_duration.append(duration)

        self.dsts_norm = []
        self.dsts_duration = []
        for traj in self.dsts:
            timestamps = np.array([item[2] for item in traj])
            t_min, t_max = timestamps.min(), timestamps.max()
            duration = t_max - t_min
            t_range = duration if duration != 0 else 1
            norm_traj = []
            norm_times = (timestamps - t_min) / t_range
            for idx, (poi, tag, ts, local_hour) in enumerate(traj):
                norm_time = norm_times[idx]
                norm_traj.append((poi, tag, ts, local_hour, norm_time, self.poi_coord_norm[poi]))
            self.dsts_norm.append(norm_traj)
            self.dsts_duration.append(duration)

        if not os.path.exists(f'../{self.args.dataset_name}/poi_id.pkl'):
            os.makedirs(os.path.dirname(f'../{self.args.dataset_name}/poi_id.pkl'), exist_ok=True)
            with open(f'../{self.args.dataset_name}/poi_id.pkl', 'wb') as f:
                pickle.dump(self.poi_idx, f)
        if not os.path.exists(f'../{self.args.dataset_name}/region_poi.pkl'):
            os.makedirs(os.path.dirname(f'../{self.args.dataset_name}/region_poi.pkl'), exist_ok=True)
            with open(f'../{self.args.dataset_name}/region_poi.pkl', 'wb') as f:
                pickle.dump(self.region_poi, f)

        
    def __getitem__(self, index):
        """
        Retrieves an item at a specified index in the dataset.
        Returns:
            tuple
        """
        uid = self.uids[index]
        o = self.oris_norm[index]
        d = self.dsts_norm[index]
        t = self.trans[index]
        ori_ck = torch.LongTensor(list(map(lambda y: y[0], o)))
        # dst_ck = torch.LongTensor(list(map(lambda y: y[0], d))).unique()
        dst_ck = torch.LongTensor(list(map(lambda y: y[0], d)))
        o_hour = torch.LongTensor(list(map(lambda y: y[3], o)))
        d_hour = torch.LongTensor(list(map(lambda y: y[3], d)))
        # change hour 0 to 24
        o_hour[o_hour == 0] = 24
        d_hour[d_hour == 0] = 24
        # timestamp
        o_t = torch.FloatTensor(list(map(lambda y: y[4], o)))
        d_t = torch.FloatTensor(list(map(lambda y: y[4], d)))
        # location
        o_l = torch.FloatTensor(list(map(lambda y: y[5], o)))
        d_l = torch.FloatTensor(list(map(lambda y: y[5], d)))
        # Mask the intermediate POI IDs and hour IDs (excluding start and end POIs)
        d_mask_indices = torch.arange(1, len(dst_ck) - 1)
        masked_d_ck = dst_ck.clone()
        masked_d_ck[d_mask_indices] = 0.0
        masked_d_h = d_hour.clone()
        masked_d_h[d_mask_indices] = 0.0
        ori_rg = t[0]
        dst_rg = t[1]
        return uid, ori_ck, dst_ck, masked_d_ck, o_hour, d_hour, masked_d_h, o_t, d_t, o_l, d_l, ori_rg, dst_rg
    
    def __len__(self):
        """
        Returns the total number of transactions (user movements) in the dataset.
        Returns:
            int
        """
        return len(self.trans)


def random_split(dataset, dataset_name, split_path, ratios=[0.8, 0.1, 0.1]):
    """
    Splits a dataset into training, validation, and testing subsets randomly.
    Returns:
        tuple
    """
    trans = dataset.trans
    trans_by_pair = defaultdict(list)
    for u, t in enumerate(trans):
        trans_by_pair[t].append(u)
    
    train_indice, valid_indice, test_indice = [], [], []

    # if os.path.exists(split_path):
    #     train_indice, valid_indice, test_indice = np.load(split_path, allow_pickle=True)
    # else:
    for t, us in trans_by_pair.items():
        us_shuf = copy(us)
        np.random.shuffle(us_shuf)
        us_len = len(us)

        train_offset = int(us_len * ratios[0])
        valid_offset = int(us_len * (ratios[0] + ratios[1]))

        train_indice.extend(us_shuf[:train_offset])
        valid_indice.extend(us_shuf[train_offset:valid_offset])
        test_indice.extend(us_shuf[valid_offset:])

    with open(f'../{dataset_name}/data_split.pkl', 'wb') as file:
        pickle.dump([train_indice, valid_indice, test_indice], file)

    return Subset(dataset, train_indice), Subset(dataset, valid_indice), Subset(dataset, test_indice) # train_indices 是训练数据的索引列表

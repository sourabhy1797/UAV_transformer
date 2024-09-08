import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import random
import yaml
import heapq
random.seed(1933)
config = yaml.safe_load(open('config.yaml'))
# Assuming the Earth's radius is 6371 km
EARTH_RADIUS_KM = 6371.0

def dijkstra(graph, start, threshold):
    distances = [360.0]*9860
    distances[start] = 0
    queue = [(0, start)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, time in zip(graph.get(current_node, {}).get('adjacent', []), graph.get(current_node, {}).get('time', [])):
            distance = current_distance + time
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    results = []
    for i in range(9860):
        if distances[i] < threshold:
            results.append(i)
    return results

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the Haversine distance.
    """
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))
    km = EARTH_RADIUS_KM * c
    return km

class CombinedLoss(torch.nn.Module):
    def __init__(self,dataset):
        super(CombinedLoss, self).__init__()
        self.weight_haversine = 0.5
        self.weight_ce = 0.5
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index = -1)
        self.nodes = torch.tensor(pd.read_csv("./data/" + dataset + "/road/node.csv", sep=',')[['lng','lat']].to_numpy()).to("cuda:" + str(config["cuda"]))

    def forward(self, logits, targets):
        # Cross-Entropy Loss
        ce_loss = self.ce_loss(logits, targets)
        pred = torch.argmax(logits, dim=1)
        mask = targets != -1
        pred = pred[mask]
        targets = targets[mask]
        # Haversine MSE Loss
        lon_pred, lat_pred = self.nodes[pred, 0], self.nodes[pred, 1]
        lon_true, lat_true = self.nodes[targets,0], self.nodes[targets,1]
        haversine_mse = torch.mean(haversine_distance(lon_pred, lat_pred, lon_true, lat_true) ** 2)

        # Combined Loss
        combined_loss = self.weight_haversine * haversine_mse + self.weight_ce * ce_loss
        return combined_loss

def load_network(dataset):
    """
    load road network from file with Pytorch geometric data object
    :param dataset: the city name of road network
    :return: Pytorch geometric data object of the graph
    """
    edge_path = "./data/" + dataset + "/road/edge_weight.csv"
    node_embedding_path = "./data/" + dataset + "/node_features.npy"

    node_embeddings = np.load(node_embedding_path)
    df_dege = pd.read_csv(edge_path, sep=',')

    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    edge_attr = df_dege["length"].to_numpy()

    edge_index = torch.LongTensor(edge_index).t().contiguous()
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    print("node embeddings shape: ", node_embeddings.shape)
    print("edge_index shape: ", edge_index.shape)
    print("edge_attr shape: ", edge_attr.shape)

    road_network = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)

    return road_network

class DataLoader():
    def __init__(self, load_part):
        self.load_part = load_part
        self.train_set = 20752
        self.vali_set = 6917
        self.test_set = 6917
        self.batch_size = config[load_part + "_batch"]
        self.node_triplets = []
        self.start = 54483

    def return_triplets_num(self):
        if self.load_part=='train':
            node_list = np.load(str(config["shuffle_node_file"]), allow_pickle=True)[:self.train_set]
        elif self.load_part=='vali':
            node_list = np.load(str(config["shuffle_node_file"]), allow_pickle=True)[self.train_set:self.train_set+self.vali_set]
        else:
            node_list = np.load(str(config["shuffle_node_file"]), allow_pickle=True)[self.train_set+self.vali_set:self.train_set+self.vali_set+self.test_set]
        
        anchor_index = list(range(len(node_list)))
        if self.load_part == 'train':
            random.shuffle(anchor_index)
        for i in anchor_index:
            node_sample = node_list[i]  # anchor sample
            self.node_triplets.append(node_sample)   # nodelist
        self.node_triplets = np.array(self.node_triplets)
        self.start = len(self.node_triplets)
        return len(self.node_triplets)

    def getbatch_one(self):
        '''
        # batch random
        index = list(range(len(self.apn_node_triplets)))
        random.shuffle(index)
        batch_index = random.sample(index, self.batch_size)

        # batch ordered
        if self.start + self.batch_size > len(self.apn_node_triplets):
            self.start = 0
        batch_index = list(range(self.start, self.start + self.batch_size))
        self.start += self.batch_size
        '''

        # batch reverse
        if self.start - self.batch_size < 0:
            self.start = len(self.node_triplets)
        batch_index = list(range(self.start - self.batch_size, self.start))
        self.start -= self.batch_size

        node_list = self.node_triplets[batch_index]

        node_batch = []

        for tri in node_list:
            node_batch.append(tri)

        return np.array(node_batch)
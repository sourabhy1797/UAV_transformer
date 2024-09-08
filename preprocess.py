import yaml
import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import Node2Vec
import random

random.seed(1953)
def prepare_dataset(trajfile):
    """
    :param trajfile: map-matching result
    """
    node_list = pd.read_csv(trajfile)
    node_list = node_list.Node_list

    node_list_int = []
    for nlist in node_list:
        tmp_list = []
        nlist = nlist[1:-1].replace('[', '').replace(']', '').replace(' ', ',').replace('\n', ',').split(',')
        for n in nlist:
            if n != '':
                tmp_list.append(int(n))
        node_list_int.append(tmp_list)

    seq_lengths = list(map(len, node_list_int))

    ## Embedding Encode
    for traj_one in node_list_int:
        traj_one += [-1]*(max(seq_lengths)-len(traj_one))
    node_list_int = np.array(node_list_int)

    shuffle_index = list(range(len(node_list_int)))
    random.shuffle(shuffle_index)

    node_list_int = node_list_int[shuffle_index]

    np.save(str(config["shuffle_node_file"]), node_list_int)

def read_graph(dataset):
    """
    Read network edages from text file and return networks object
    :param file: input dataset name
    :return: edage index with shape (n,2)
    """
    dataPath = "./data/" + dataset
    edge = dataPath + "/road/edge_weight.csv"
    node = dataPath + "/road/node.csv"

    df_dege = pd.read_csv(edge, sep=',')
    df_node = pd.read_csv(node, sep=',')

    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    num_node = df_node["node"].size

    print("{0} road netowrk has {1} edages.".format(config["dataset"], edge_index.shape[0]))
    print("{0} road netowrk has {1} nodes.".format(config["dataset"], num_node))

    return edge_index, num_node
    
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch(model, loader, optimizer):
    # Training with epoch iteration
    last_loss = 1
    print("Training node embedding with node2vec...")
    for i in range(100):
        loss = train(model, loader, optimizer)
        print('Epoch: {0} \tLoss: {1:.4f}'.format(i, loss))
        if abs(last_loss - loss) < 1e-5:
            break
        else:
            last_loss = loss

@torch.no_grad()
def save_embeddings(model, num_nodes, dataset, device):
    model.eval()
    node_features = model(torch.arange(num_nodes, device=device)).cpu().numpy()
    np.save("./data/" + dataset + "/node_features.npy", node_features)
    
    np.savetxt("./data/" + dataset + "/node_features.csv", node_features, delimiter=',')

    print("Node embedding saved at: ./data/" + dataset + "/node_features.npy")
    return

if __name__ == "__main__":
    config = yaml.safe_load(open('config.yaml'))
    edge_index, num_node = read_graph(str(config["dataset"]))

    device = "cuda:" + str(config["cuda"])
    embedding_size = config["embedding_size"]
    walk_length = config["node2vec"]["walk_length"]
    context_size = config["node2vec"]["context_size"]
    walks_per_node = config["node2vec"]["walks_per_node"]
    p = config["node2vec"]["p"]
    q = config["node2vec"]["q"]

    edge_index = torch.LongTensor(edge_index).t().contiguous().to(device)
    print(edge_index.shape)
    model = Node2Vec(
        edge_index,
        embedding_dim=embedding_size,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        p=p,
        q=q,
        sparse=True,
        num_nodes=num_node
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    # Train until delta loss has been reached
    train_epoch(model, loader, optimizer)
    save_embeddings(model, num_node, str(config["dataset"]), device)
    prepare_dataset(trajfile=str(config["traj_file"]))
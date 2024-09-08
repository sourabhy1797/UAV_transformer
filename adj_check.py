import pandas as pd
import pickle
from tqdm import tqdm
from data_utils import dijkstra

def prepare_adj(dataset):
    df_dege = pd.read_csv("./data/" + dataset + "/road/edge_weight.csv", sep=',')
    df_node = pd.read_csv("./data/" + dataset + "/road/node.csv", sep=',')
    graph = {}
    ## Construct graph
    for index, row in df_dege.iterrows():
        s_node, e_node, length, speed = int(row['s_node']), int(row['e_node']), row['length'], row['max_speed']
        time = length / (speed*0.44704)  # Calculate time in s
        ## store starting node and ending node
        if s_node not in graph:
            graph[s_node] = {'adjacent': [], 'time': []}
        graph[s_node]['adjacent'].append(e_node)
        graph[s_node]['time'].append(time)
        if e_node not in graph:
            graph[e_node] = {'adjacent': [], 'time': []}
        graph[e_node]['adjacent'].append(s_node)
        graph[e_node]['time'].append(time)

    # print(max_time) # 160
    adjacency_dict = {}
    for node in tqdm(df_node["node"]):
        adjacency_dict[node] = dijkstra(graph, node, 25) # threshold

    with open(f'model/{dataset}/adjacency_set.pkl', 'wb') as f:
        pickle.dump(adjacency_dict, f)

def check_trajectory(trajectory, adjacency_dict):
    """
    检查给定轨迹中的每个点是否可达。
    
    :param trajectory: 由节点ID组成的序列
    :param adjacency_dict: 邻接集合字典
    :return: 轨迹的可达性验证结果的列表
    """
    validity_list = []
    for i in range(len(trajectory) - 1):
        current_node = trajectory[i]
        next_node = trajectory[i + 1]
        # 检查下一个点是否在当前点的邻接集合中
        if next_node in adjacency_dict.get(current_node, set()):
            validity_list.append(True)
        else:
            validity_list.append(False)
    return validity_list

if __name__ == "__main__":
    # Regenerate adjacency_set.pkl
    prepare_adj('sanfrancisco')

    # 载入之前保存的邻接集合
    with open('model/sanfrancisco/adjacency_set.pkl', 'rb') as f:
        adjacency_dict = pickle.load(f)

    trajectory_example = [7958, 496, 7689, 8154, 512, 511, 7943, 9353, 9324, 9319, 9066, 4945, 7584, 6609, 1164, 2034, 9564, 3226, 3227, 6738]
    trajectory_validity = check_trajectory(trajectory_example, adjacency_dict)


    for node, is_valid in zip(trajectory_example[:-1], trajectory_validity):
        print(f"From node {node} to next is valid: {is_valid}")



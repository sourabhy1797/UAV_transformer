import csv
import random
import math
import pickle
import numpy as np
# 设置随机种子以确保结果可复现
random.seed(500)
# 从CSV文件加载节点信息
def load_node_info(csv_file):
    nodes_dict = {}
    with open(csv_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            node_id = int(row['node'])
            lng = float(row['lng'])
            lat = float(row['lat'])
            nodes_dict[node_id] = (lng, lat)
    return nodes_dict

# 计算哈弗辛距离
def haversine_distance(coord1, coord2):
    R = 6371  # 地球半径，单位公里
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # 输出结果单位为公里
    return distance

# 计算单个fake节点到多个POI的utility loss
def compute_utility_loss_single_fake(nodes_dict, real_node, fake_node, pois):
    total_loss = 0
    for poi in pois:
        distance_real_poi = haversine_distance(nodes_dict[real_node], nodes_dict[poi])
        distance_fake_poi = haversine_distance(nodes_dict[fake_node], nodes_dict[poi])
        total_loss += abs(distance_real_poi - distance_fake_poi)
    return total_loss

# Load adjacency information from pickle file
with open('model/rome/adjacency_set.pkl', 'rb') as f:
    adjacency_dict = pickle.load(f)

# Assume trajectory_example is the sequence of real nodes provided earlier
trajectory_example = [11837, 21794, 11830, 21794, 11824, 24974, 23372, 11820, 24974, 21800, 11815, 25872, 323, 20153, 1515, 19930, 1513, 19930, 19929, 1517, 1628, 33010, 1561, 1562, 18075, 1562, 18075, 18079, 1538, 1569]
 # Your real node sequence

# 从CSV文件加载节点信息
nodes_dict = load_node_info('data/rome/road/node.csv')
pois = random.sample(list(nodes_dict.keys()), 100)

# 创建一个形状为(trajectory_length, 9860)的数组，初始填充为0
trajectory_length = len(trajectory_example)
utility_loss_array = np.zeros((trajectory_length, 9860))

# 第一个real_node的所有fake_nodes默认为0，所以从第二个real_node开始计算
for i in range(1, trajectory_length):
    previous_node = trajectory_example[i - 1]
    current_node = trajectory_example[i]
    # 确定当前real_node的location set中的fake_nodes，基于上一个real_node的reachable nodes
    if previous_node in adjacency_dict:
        previous_reachable_nodes = set(adjacency_dict[previous_node])
        for fake_node in range(9860):
            if fake_node in previous_reachable_nodes:
            # 计算utility loss
                utility_loss = compute_utility_loss_single_fake(nodes_dict, current_node, fake_node, pois)
                # 如果utility loss不为0，取其倒数
                if utility_loss != 0:
                    utility_loss_array[i][fake_node] = 10/ utility_loss
            # 如果不可达或utility loss为0，则保持数组中的值为0

print(utility_loss_array[5])
print(utility_loss_array.shape)

for i in range(trajectory_length):
    # 找到第一个非零元素
    non_zero_elements = utility_loss_array[i][utility_loss_array[i] != 0]
    if len(non_zero_elements) > 0:  # 如果存在非零元素
        first_non_zero = non_zero_elements[0]  # 获取第一个非零元素
        print(f"第 {i} 个real_node的第一个非零utility loss的倒数是: {first_non_zero}")
    else:
        print(f"第 {i} 个real_node没有非零utility loss值。")
# # 将排序后的结果写入新的CSV文件
# with open('locationset_filtered_new.csv', mode='w', newline='') as file:
#     csv_writer = csv.writer(file)
#     csv_writer.writerow(['Node', 'Location set Nodes'])
    
#     for real_node, sorted_losses in sorted_results:
#         # 将假节点和它们的loss转换为字符串
#         sorted_fake_nodes_str = ','.join([str(node) for node, _ in sorted_losses])
#         csv_writer.writerow([real_node, sorted_fake_nodes_str])

# # 检查节点"5"是否在邻接字典中
# if 5 in adjacency_dict:
#     # 获取节点"5"的邻接信息
#     adjacency_info = adjacency_dict[1]
#     # 打印节点"5"的邻接节点和对应的时间
#     print(f"Node 5 adjacent nodes: {adjacency_info}")
# else:
#     print("Node 5 is not in the adjacency dictionary.")

# import pandas as pd
# import pickle
import pickle

# # 载入之前保存的邻接集合
# with open('model/rome/adjacency_set.pkl', 'rb') as f:
#     adjacency_dict = pickle.load(f)

# def check_trajectory(trajectory, adjacency_dict):
#     """
#     检查给定轨迹中的每个点是否可达。
    
#     :param trajectory: 由节点ID组成的序列
#     :param adjacency_dict: 邻接集合字典
#     :return: 轨迹的可达性验证结果的列表
#     """
#     validity_list = []
#     for i in range(len(trajectory) - 1):
#         current_node = trajectory[i]
#         next_node = trajectory[i + 1]
#         # 检查下一个点是否在当前点的邻接集合中
#         if next_node in adjacency_dict.get(current_node, set()):
#             validity_list.append(True)
#         else:
#             validity_list.append(False)
#     return validity_list

# # 示例轨迹
# trajectory_example = [11837, 21794, 11830, 21794, 11824, 24974, 23372, 11820, 24974, 21800, 11815, 25872, 323, 20153, 1515, 19930, 1513, 19930, 19929, 1517, 1628, 33010, 1561, 1562, 18075, 1562, 18075, 18079, 1538, 1569]
# trajectory_validity = check_trajectory(trajectory_example, adjacency_dict)

# # 打印验证结果
# for node, is_valid in zip(trajectory_example[:-1], trajectory_validity):
#     print(f"From node {node} to next is valid: {is_valid}")

# 载入之前保存的邻接集合
with open('model/rome/adjacency_set.pkl', 'rb') as f:
    adjacency_dict = pickle.load(f)

# 指定要查看邻接集合的节点
node_id = 18429

# 打印出指定节点的邻接集合
if node_id in adjacency_dict:
    print(f"Adjacency set of node {node_id}: {adjacency_dict[node_id]}")
else:
    print(f"No adjacency set found for node {node_id}")



# # 载入之前保存的邻接集合
# with open('model/rome/adjacency_set.pkl', 'rb') as f:
#     adjacency_dict = pickle.load(f)

# def is_trajectory_valid(trajectory, adjacency_dict):
#     """
#     判断轨迹是否有效（所有点都可达）
#     """
#     for i in range(len(trajectory) - 1):
#         if trajectory[i + 1] not in adjacency_dict.get(trajectory[i], set()):
#             return False
#     return True

# # 读取轨迹数据集
# trajectories_df = pd.read_csv('all_possible_trajectories.csv', header=None)

# # 检查每个轨迹
# valid_trajectories = []
# for _, trajectory in trajectories_df.iterrows():
#     trajectory = trajectory.tolist()
#     if is_trajectory_valid(trajectory, adjacency_dict):
#         valid_trajectories.append(trajectory)

# # 将有效的轨迹保存到新的CSV文件中
# valid_trajectories_df = pd.DataFrame(valid_trajectories)
# valid_trajectories_df.to_csv('filtered_trajectories.csv', index=False, header=False)

# print(f"Saved {len(valid_trajectories)} valid trajectories to 'filtered_trajectories.csv'.")

# # 优化后的代码
# import pandas as pd
# import pickle

# csv_file_path = 'data/rome/st_traj/tree.csv'

# # 读取CSV文件
# df = pd.read_csv(csv_file_path)

# # 初始化邻接集合字典
# adjacency_dict = {}

# # 假设第一列是节点编号
# for index, row in df.iterrows():
#     node = row['Node']
#     weights = map(float, row['shortest_weight'].split(','))
#     # 确保每个节点的邻接节点都被存储在一个集合中
#     adjacency_dict[node] = set()
#     for adj_node, weight in enumerate(weights):
#         if weight < 30.0:
#             adjacency_dict[node].add(adj_node)  # 存储邻接节点

# # 存储邻接集合到文件
# with open('adjacency_set.pkl', 'wb') as f:
#     pickle.dump(adjacency_dict, f)

# print("Adjacency set saved successfully!")
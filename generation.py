
# import pandas as pd
# import csv

# # 读取 test.csv 文件
# with open('data/rome/st_traj/test_data.csv', 'r') as file:
#     # 创建一个空的 DataFrame 来存储满足条件的轨迹
#     frames = []
#     count = 0  # 计数器，用于限制处理的轨迹数量

#     for line in file:
#         nodes = line.strip().split(',')
#         if len(nodes) > 10:
#             # 如果轨迹长度大于 10，将其添加到 DataFrame 列表中
#             frame = pd.DataFrame({'轨迹': [','.join(nodes)]})
#             frames.append(frame)
#             count += 1

#             if count >= 50:  # 达到处理的轨迹数量限制
#                 break

# # 合并所有满足条件的轨迹
# filtered_trails = pd.concat(frames, ignore_index=True)

# # 保存满足条件的轨迹到 data_sample.csv，禁用引号包裹
# filtered_trails.to_csv('data_sample.csv', index=False, header=False, mode='w')
# # filtered_trails.to_csv('data_sample.csv', index=False, header=False, mode='w', quoting=csv.QUOTE_NONE, escapechar='\\')

import pandas as pd
import folium
import csv
import re

# 读取节点信息
node_df = pd.read_csv('data/rome/road/node.csv')

# 创建一个从节点 ID 到坐标的字典
node_dict = {row['node']: (row['lat'], row['lng']) for index, row in node_df.iterrows()}

# 函数：根据节点 ID 获取经纬度
def get_coords(node_id):
    # 从之前创建的字典 node_dict 获取坐标
    return node_dict.get(node_id)

# 准备不同颜色的轨迹颜色列表
colors = [
    'blue', 'red', 'green', 'purple', 'orange', 'pink', 'yellow', 'brown', 'gray', 'cyan',
    'magenta', 'lime', 'indigo', 'violet', 'black', 'beige', 'olive', 'navy', 'teal', 'maroon'
]

# 初始化地图
m = folium.Map(location=[41.9028, 12.4964], zoom_start=12)

# 直接读取文件行
with open('data_generation.csv', 'r') as file:
    for idx, line in enumerate(file):
        # 使用正则表达式提取数字
        nodes = re.findall(r'\d+', line)

        # 转换为整数列表
        nodes = [int(n) for n in nodes]

        coords = [get_coords(n) for n in nodes if get_coords(n) is not None]

        # 打印信息以调试
        print(f"Line {idx}: Coords = {coords}")

        # 绘制轨迹线
        if coords:
            color = colors[idx % len(colors)]
            folium.PolyLine(coords, color=color, weight=2.5, opacity=1).add_to(m)
            for coord in coords:
                folium.CircleMarker(location=coord, radius=3, color=color, fill=True, fill_color=color).add_to(m)

m.save('trajectory_map.html')

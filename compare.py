import folium
import pandas as pd

# 读取数据
nodes_df = pd.read_csv('data/rome/road/node.csv', index_col='node')
locationsets_df = pd.read_csv('locationset_filtered_ori.csv')
# 创建地图对象
m = folium.Map(location=[41.9028, 12.4964], zoom_start=11, tiles='CartoDB Positron')

# 创建一组颜色，用于不同的location sets
import matplotlib.colors as mcolors
colors = list(mcolors.CSS4_COLORS.values())  # 使用matplotlib的颜色

# 绘制轨迹节点并用不同颜色标记每个节点的location set中的前4个点
for index, row in locationsets_df.iterrows():
    # 获取轨迹节点信息
    traj_node = int(row['Node'])
    traj_node_coords = nodes_df.loc[traj_node, ['lat', 'lng']].values.tolist()
    
    # 添加轨迹节点
    folium.CircleMarker(traj_node_coords, radius=7, color='black', fill=True).add_to(m)
    
    # 获取location set中的前4个点
    location_set = [int(x) for x in row['location set Nodes for each step(Aligned to min size: 204)'].split(',')[1:8]]
    set_color = colors[index % len(colors)]  # 循环使用颜色列表中的颜色
    
    # 为location set中的每个节点添加相同颜色的标记
    for loc_node in location_set:
        loc_node_coords = nodes_df.loc[loc_node, ['lat', 'lng']].values.tolist()
        folium.CircleMarker(loc_node_coords, radius=10, color=set_color, fill=True).add_to(m)

# 绘制轨迹线
trajectory_nodes = locationsets_df['Node'].astype(int).tolist()
trajectory_coords = nodes_df.loc[trajectory_nodes, ['lat', 'lng']].values.tolist()
folium.PolyLine(trajectory_coords, color='red', weight=5.5, opacity=0.7).add_to(m)

# 保存地图到 HTML 文件
m.save('ori_map.html')
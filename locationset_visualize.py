import folium
import pandas as pd

# 读取数据
nodes_df = pd.read_csv('data/rome/road/node.csv', index_col='node')
# locationsets_df = pd.read_csv('locationset_filtered_new.csv')
locationsets_df = pd.read_csv('locationset.csv')


# 创建地图对象
m = folium.Map(location=[41.9028, 12.4964], zoom_start=12, tiles='CartoDB Positron')

# 手动指定颜色列表
colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred']

# 绘制完整的轨迹线
trajectory_nodes = locationsets_df['Node'].astype(int).tolist()
trajectory_coords = nodes_df.loc[trajectory_nodes, ['lat', 'lng']].values.tolist()
folium.PolyLine(trajectory_coords, color='red', weight=5.5, opacity=0.7).add_to(m)

x = 4  # 起始索引
y = 5  # 结束索引
for index, node_id in enumerate(trajectory_nodes[x:y+1]):
    node_coords = nodes_df.loc[node_id, ['lat', 'lng']].values.tolist()
    set_color = colors[index % len(colors)]
    popup_text = f"Location Set Color: {set_color}"
    folium.CircleMarker(
        node_coords, 
        radius=7, 
        color='black', 
        fill=True, 
        popup=popup_text
    ).add_to(m)

for index, row in locationsets_df.iterrows():
    # 检查当前索引是否在x和y之间
    if x <= index <= y:
        location_set = [int(x) for x in row['location set Nodes for each step(Aligned to min size: 204)'].split(',')[1:11]]
        set_color = colors[index % len(colors)]
        
        # 为location set中的每个节点添加相同颜色的标记
        for loc_node in location_set:
            loc_node_coords = nodes_df.loc[loc_node, ['lat', 'lng']].values.tolist()
            folium.CircleMarker(
                loc_node_coords, 
                radius=10, 
                color=set_color, 
                fill=True
            ).add_to(m)

# 保存地图到 HTML 文件
m.save('map_ori.html')

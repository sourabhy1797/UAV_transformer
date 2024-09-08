import pandas as pd
import folium

def visualize_trajectory_from_x_to_y(data_file, x, y):
    # 读取数据集
    data = pd.read_csv(data_file)

    # 创建一个初始地图
    m = folium.Map(location=[41.85732470000001, 12.4383377], zoom_start=14)

    # 定义一个颜色列表
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
        'gray', 'black', 'lightgray', 'yellow'
    ]

    # 从第x条到第y条轨迹
    for index, row in data.iloc[x-1:y].iterrows():
        coor_list = eval(row['Coor_list'])
        node_list = eval(row['Node_list'])
        # 纠正经纬度
        corrected_coor_list = [(lat, lon) for lon, lat in coor_list]

        # 在地图上添加轨迹
        folium.PolyLine(corrected_coor_list, color=colors[index%20], weight=2.5, opacity=1).add_to(m)

        # 在地图上添加每个节点点并显示轨迹ID和节点ID
        for coor, node_id in zip(corrected_coor_list, node_list):
            folium.CircleMarker(
                location=coor,
                radius=5,
                color=colors[index%20],  # 使用index%20确保颜色列表不会越界
                fill=True,
                fill_color=colors[index%20],
                popup=f"Traj_id: {row['Traj_id']}\nNode_id: {node_id}"
            ).add_to(m)

    # 保存地图到HTML文件，或直接显示
    output_file_name = f"visualize/map_from_{x}_to_{y}.html"
    m.save(output_file_name)
    return output_file_name

# 示例：从第3条轨迹到第8条轨迹
file_name = visualize_trajectory_from_x_to_y("data/rome/st_traj/cleaned_data.csv", 800, 1600)
print(f"Map saved to {file_name}")

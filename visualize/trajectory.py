import pandas as pd
import folium
import ast
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from folium import DivIcon

mapbox_access_token = 'pk.eyJ1IjoiMTIzeHhwIiwiYSI6ImNscDFzMmRwajBqeWYybHM0cjRnNWpwb3IifQ.4NDwGATIws7GYUNm2WlTXg'
tiles = 'https://api.mapbox.com/styles/v1/123xxp/clp68rhtj011g01pe2y670y03/tiles/256/{z}/{x}/{y}@2x?access_token=' + mapbox_access_token

# 创建固定数量颜色的渐变
def create_fixed_color_gradient(n_colors, cmap):
    colormap = cm.get_cmap(cmap, n_colors)
    return [mcolors.rgb2hex(colormap(i)) for i in range(n_colors)]

# 根据分数生成颜色
def get_color(score, min_score, max_score):
    return (score - min_score) / (max_score - min_score)

# 加载数据
def load_data(file_path, target_node):
    df = pd.read_csv(file_path)
    row = df[df['Real_Node'] == target_node].iloc[0]
    return ast.literal_eval(row['Filtered_Location_Set']), ast.literal_eval(row['Scores'])

df = pd.read_csv('data/rome/road/node.csv')
real_node_target = 2557

# 预定义的轨迹连线节点
real_trajectory = [18429, 14796, 2924, 18179, 2923, 2498, 21058, 7099, 2935, 2061, 2556, 2557, 2490, 7910, 2491, 2492, 7875, 8757, 8740, 2650, 2675]

# Choosing Colormap
datasets = {
    'output/location_setWeight0.1.csv.csv': 'magma',

    'output/location_setWeight0.01.csv.csv': 'viridis'
}

for data_path, colormap in datasets.items():
    location_set, scores = load_data(data_path, real_node_target)
    min_score, max_score = min(scores), max(scores)
    color_bar = create_fixed_color_gradient(200, colormap)
    color_gradient = cm.get_cmap(colormap)
    colors_for_points = [mcolors.to_hex(color_gradient(get_color(score,min_score, max_score))) for score in scores]
    print(colors_for_points)

    # 创建地图实例
    first_node = df[df['node'] == real_node_target].iloc[0]
    m = folium.Map(location=[first_node['lat'], first_node['lng']], tiles=tiles, attr='Mapbox', zoom_start=12)

    # 添加点
    for node_id, color in zip(location_set, colors_for_points):
        node = df[df['node'] == node_id].iloc[0]
        folium.CircleMarker(location=[node['lat'], node['lng']], radius=7, color=color, fill=True, fill_color=color, fill_opacity=1).add_to(m)

    # 添加轨迹连线
    second_group_coords = []
    for node_id in real_trajectory:
        node_coords = df[df['node'] == node_id]
        if not node_coords.empty:
            second_group_coords.append([node_coords.iloc[0]['lat'], node_coords.iloc[0]['lng']])
            node_color = 'grey' if node_id != real_node_target else 'red'
            node_radius = 6 if node_id != real_node_target else 8
            folium.CircleMarker(
                location=[node_coords.iloc[0]['lat'], node_coords.iloc[0]['lng']],
                radius=node_radius,
                color=node_color,
                fill=True,
                fill_color=node_color,
                fill_opacity=0.8
            ).add_to(m)

    folium.PolyLine(second_group_coords, color='grey', weight=5, opacity=0.81).add_to(m)

    # 添加颜色条
    color_bar_html = "<div style='width: 25px; height: 230px; position: relative; font-size: 10px; text-align: center;'><div style='position: absolute; top: 0; left: 0; right: 0;'>{max}</div><div style='position: absolute; top: 20px; height: 190px; width: 100%; background: linear-gradient(to top, {colors});'></div><div style='position: absolute; bottom: 0; left: 0; right: 0;'>{min}</div></div>".format(max="{:.3f}".format(max_score), colors=', '.join(color_bar), min="{:.3f}".format(min_score))
    lon, lat = 12.5457755, 41.93528934
    folium.Marker(location=[lat, lon], icon=DivIcon(icon_size=(150, 350), icon_anchor=(0, 0), html=color_bar_html)).add_to(m)

    # 保存地图为HTML文件
    output_file_name = data_path.split('/')[-1].split('.c')[0] + '.html'
    m.save(output_file_name)

import pandas as pd
import folium
import numpy as np

mapbox_access_token = 'pk.eyJ1IjoiMTIzeHhwIiwiYSI6ImNscDFzMmRwajBqeWYybHM0cjRnNWpwb3IifQ.4NDwGATIws7GYUNm2WlTXg'
tiles = 'https://api.mapbox.com/styles/v1/123xxp/clp68rhtj011g01pe2y670y03/tiles/256/{z}/{x}/{y}@2x?access_token=' + mapbox_access_token

# 定义四条轨迹
original_trajectory = [18429, 14796, 2924, 18179, 2923, 2498, 21058, 7099, 2935, 2061, 2556, 2557, 2490, 7910, 2491, 2492, 7875, 8757, 8740, 2650, 2675]
obfuscated_trajectory = [2557, 2495, 2493, 18166, 2494, 18170, 23672, 18168, 18167, 30935, 7629, 23673]
vehitrack_phase1_trajectory = [22627, 30281, 2107, 2556, 18167, 2557, 2495, 18166, 2493, 2494, 18170, 22646, 18168, 2062, 23672, 23673]
vehitrack_rnn_trajectory = [22627, 30281, 2107, 2556, 18167, 2557, 2495, 18166, 2493, 2494, 18170, 22646, 18168, 2062, 23672, 23673]

# 定义四种颜色
colors = ['blue', 'green', 'red', 'purple']
# 读取节点数据
df = pd.read_csv('data/rome/road/node.csv')
def add_trajectory_to_map(trajectory, color, map_object, draw_line=True, shape='circle'):
    coords = []
    for node_id in trajectory:
        node_coords = df[df['node'] == node_id]
        if not node_coords.empty:
            lat, lng = node_coords.iloc[0]['lat'], node_coords.iloc[0]['lng']
            coords.append([lat, lng])
            # 根据形状添加标记
            if shape == 'circle':
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(map_object)
            elif shape == 'triangle':
                folium.RegularPolygonMarker(
                    location=[lat, lng],
                    number_of_sides=3,
                    radius=6,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(map_object)
            elif shape == 'square':
                folium.RegularPolygonMarker(
                    location=[lat, lng],
                    number_of_sides=4,
                    radius=6,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(map_object)
            elif shape == 'pentagon':
                folium.RegularPolygonMarker(
                    location=[lat, lng],
                    number_of_sides=5,
                    radius=6,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(map_object)
    # 如果需要，添加轨迹的连线
    if draw_line:
        folium.PolyLine(coords, color=color, weight=4, opacity=0.7).add_to(map_object)

# 创建地图实例
m = folium.Map(location=[41.9096634, 12.5008688], tiles=tiles, attr='Mapbox', zoom_start=12)

# 添加轨迹和节点，指定形状参数
add_trajectory_to_map(original_trajectory, colors[0], m, draw_line=True, shape='circle')
add_trajectory_to_map(obfuscated_trajectory, colors[1], m, draw_line=False, shape='triangle')
add_trajectory_to_map(vehitrack_phase1_trajectory, colors[2], m, draw_line=False, shape='pentagon')  # 使用五边形
add_trajectory_to_map(vehitrack_rnn_trajectory, colors[3], m, draw_line=False, shape='square')

# 创建一个特征组来存放标注信息
# 创建一个特征组来存放标注信息
legend = folium.FeatureGroup()

# 定义标注内容，不包含“Legend”标题
legend_info = '''
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 265px; height: 120px; 
     border:2px solid grey; z-index:9999; font-size:17px; font-weight:bold;
     ">
    &nbsp; <span style="color:blue; font-size: 17px">-●-</span> Original <br>
    &nbsp; <span style="color:green; font-size: 15px;">▲</span> Obfuscated <br>
    &nbsp; <span style="color:red; font-size: 14px;">⬟</span> VehiTrack (after Phase One) <br>
    &nbsp; <span style="color:purple; font-size: 15px;">&#9632;</span> VehiTrack (after Phase Two)
</div>


'''

# 将标注信息添加到特征组
legend.add_child(folium.Marker(
    location=[41.910606634, 12.4700688],  # 这个位置是地图上的点，将其设置为不可见
    icon=folium.DivIcon(
        html=legend_info
    )
))

# 将特征组添加到地图上
m.add_child(legend)

# 保存地图为HTML文件
m.save('trajectories_map1.html')

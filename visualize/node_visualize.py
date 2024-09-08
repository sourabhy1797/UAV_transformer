import pandas as pd
import folium
import ast
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from folium import DivIcon

mapbox_access_token = 'pk.eyJ1IjoiMTIzeHhwIiwiYSI6ImNscDFzMmRwajBqeWYybHM0cjRnNWpwb3IifQ.4NDwGATIws7GYUNm2WlTXg'
tiles = 'https://api.mapbox.com/styles/v1/123xxp/clp68rhtj011g01pe2y670y03/tiles/256/{z}/{x}/{y}@2x?access_token=' + mapbox_access_token

# 创建颜色渐变
def get_heatmap_gradient(n, colormap):
    return [mcolors.rgb2hex(colormap(i / n)) for i in range(n-1, -1, -1)]
import matplotlib.pyplot as plt

def create_fixed_color_gradient(n_colors):
    # 使用新的方法来获取颜色映射
    colormap = plt.colormaps['viridis'](np.linspace(0, 1, n_colors))
    return [mcolors.rgb2hex(colormap[i]) for i in range(n_colors)]

# 根据分数创建颜色
def get_color(score, min_score, max_score, colormap):
    normalized_score = (score - min_score) / (max_score - min_score)
    return mcolors.rgb2hex(colormap(normalized_score))

# 非线性分数调整
def adjust_score(score, min_score, max_score):
    shifted_score = score - min_score + 1
    return np.log(shifted_score)

# 分数归一化
def normalize_scores(scores, min_score, max_score):
    shifted_scores = [score - min_score + 1 for score in scores]
    log_scores = np.log(shifted_scores)
    min_log_score, max_log_score = np.log(1), np.log(max_score - min_score + 1)
    return (log_scores - min_log_score) / (max_log_score - min_log_score)

# 读取数据
def load_data(file_path, target_node):
    df = pd.read_csv(file_path)
    row = df[df['Real_Node'] == target_node]
    if row.empty:
        print(f"No data found for Real_Node {target_node}")
        return None, None
    return ast.literal_eval(row['Filtered_Location_Set'].iloc[0]), ast.literal_eval(row['Scores'].iloc[0])

# 创建一个热图风格的颜色渐变函数
def get_heatmap_gradient(n, colormap):
    return [mcolors.rgb2hex(colormap(i / n)) for i in range(n-1, -1, -1)]



mapbox_access_token = 'pk.eyJ1IjoiMTIzeHhwIiwiYSI6ImNscDFzMmRwajBqeWYybHM0cjRnNWpwb3IifQ.4NDwGATIws7GYUNm2WlTXg'
tiles = 'https://api.mapbox.com/styles/v1/123xxp/clp68rhtj011g01pe2y670y03/tiles/256/{z}/{x}/{y}@2x?access_token=' + mapbox_access_token

# 读取CSV文件
df = pd.read_csv('data/rome/road/node.csv')
data_df = pd.read_csv('output/location_setWeight0.1.csv.csv')

# 2490
real_node_target = 2490
row = data_df[data_df['Real_Node'] == real_node_target]

if not row.empty:
    # 解析Filtered_Location_Set列
    filtered_location_set_str = row['Filtered_Location_Set'].iloc[0]
    location_set = ast.literal_eval(filtered_location_set_str)
    # 解析Scores列以获取分数列表
    scores_str = row['Scores'].iloc[0]
    scores = ast.literal_eval(scores_str)
else:
    print(f"No data found for Real_Node {real_node_target}")
    location_set = []
    scores = []

# 分数的最大值和最小值，用于归一化
min_score, max_score = min(scores), max(scores)

real_trajectory = [18429, 14796, 2924, 18179, 2923, 2498, 21058, 7099, 2935, 2061, 2556, 2557, 2490, 7910, 2491, 2492, 7875, 8757, 8740, 2650, 2675]
# 调整颜色映射的使用方式（非线性映射）
# 非线性映射函数，比如使用对数映射
def adjust_score(score, min_score, max_score):
    # 对分数进行平移和缩放，确保所有分数为正
    shifted_score = score - min_score + 1
    # 应用对数映射
    return np.log(shifted_score)
# 用于调整分数并归一化的函数
def adjust_and_normalize_scores(scores, min_score, max_score):
    # 将分数进行平移和缩放，确保所有分数为正
    shifted_scores = [score - min_score + 1 for score in scores]
    # 应用对数映射并归一化
    log_scores = np.log(shifted_scores)
    min_log_score = np.log(1)
    max_log_score = np.log(max_score - min_score + 1)
    normalized_scores = (log_scores - min_log_score) / (max_log_score - min_log_score)
    return normalized_scores
# 根据分数获取颜色
def get_color_based_on_score(score, min_score, max_score, colormap):
    adjusted_score = adjust_score(score, min_score, max_score)
    # 将调整后的分数映射到0-1范围
    normalized_score = (adjusted_score - np.log(1)) / (np.log(max_score - min_score + 1) - np.log(1))
    return mcolors.rgb2hex(colormap(normalized_score))

# 为location_set创建基于分数的颜色渐变
score_based_heatmap_gradient = [get_color_based_on_score(score, min_score, max_score, cm.plasma) for score in scores]

# 找出第一个点的坐标
first_node_id = location_set[0]
first_node = df[df['node'] == first_node_id].iloc[0]

m = folium.Map(location=[first_node['lat'], first_node['lng']], tiles=tiles, attr='Mapbox', zoom_start=12)
# 创建固定颜色渐变
n_colors = len(scores)
color_gradient = create_fixed_color_gradient(n_colors)

# 根据分数排序并分配颜色
sorted_indices = np.argsort(scores)[::-1]  # 分数从高到低排序的索引
colors_for_points = [color_gradient[idx] for idx in sorted_indices]

# 重新映射颜色到原始顺序
colors_in_original_order = [None] * len(scores)
for original_idx, sorted_idx in enumerate(sorted_indices):
    colors_in_original_order[sorted_idx] = colors_for_points[original_idx]

# 为第一组点添加循环
df_group = df[df['node'].isin(location_set)]
for idx, row in enumerate(df_group.itertuples()):
    node_color = colors_in_original_order[idx]

    folium.CircleMarker(
        location=[row.lat, row.lng],
        radius=7,
        color=node_color,
        fill=True,
        fill_color=node_color,
        fill_opacity=1,
        popup=str(row.node)
    ).add_to(m)
# 为第二组点添加循环
second_group_coords = []
for node_id in real_trajectory:
    node_coords = df[df['node'] == node_id]
    if not node_coords.empty:
        second_group_coords.append([node_coords.iloc[0]['lat'], node_coords.iloc[0]['lng']])
        node_color = 'grey' if node_id != 2490 else 'red'
        node_radius = 6 if node_id != 2490 else 8

        folium.CircleMarker(
            location=[node_coords.iloc[0]['lat'], node_coords.iloc[0]['lng']],
            radius=node_radius,
            color=node_color,
            fill=True,
            fill_color=node_color,
            fill_opacity=0.8,
            popup=str(node_id)
        ).add_to(m)
#
folium.PolyLine(second_group_coords, color='grey', weight=5, opacity=0.81).add_to(m)
# 创建颜色条的HTML
# 创建颜色条的HTML
num_points_in_color_bar = 100
color_bar_colors = []
# 归一化调整后的分数
normalized_scores = adjust_and_normalize_scores(scores, min_score, max_score)
# 使用cm.jet颜色映射
color_map = cm.jet
# 为了更精确地反映分数分布，我们可以使用分位数而不是线性插值
percentiles = np.linspace(0, 100, num_points_in_color_bar)
for percentile in percentiles:
    # 计算每个百分位的分数值
    score_at_percentile = np.percentile(normalized_scores, percentile)
    # 将这个分数映射到颜色
    color = mcolors.rgb2hex(color_map(score_at_percentile))
    color_bar_colors.append(color)
# 创建颜色条的HTML
color_bar_html = """
<div style='width: 25px; height: 230px; position: relative; font-size: 10px; text-align: center;'>
    <div style='position: absolute; top: 0; left: 0; right: 0;'> {}</div>
    <div style='position: absolute; top: 20px; height: 190px; width: 100%; background: linear-gradient(to top, {});'></div>
    <div style='position: absolute; bottom: 0; left: 0; right: 0;'> {}</div>
</div>
""".format("{:.3}".format(max_score), ', '.join(color_bar_colors), "{:.3}".format(min_score))

# 添加颜色条到地图
color_bar = DivIcon(
    icon_size=(150, 350),  # 更新标记的大小以适应新的颜色条尺寸
    icon_anchor=(0, 0),
    html=color_bar_html
)
lon, lat = 12.5457755,41.93528934
folium.Marker(location=[lat, lon], icon=color_bar).add_to(m)

# 保存为HTML文件
m.save('map_x0.1_KALL.html')
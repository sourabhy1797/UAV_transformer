import pandas as pd
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径（公里）
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

def compute_avg_distance_for_first_20(data_file):
    data = pd.read_csv(data_file).head(50000)  # 仅选取前20条轨迹
    
    total_distance = 0
    total_pairs = 0
    
    for _, row in data.iterrows():
        coor_list = eval(row['Coor_list'])
        
        for i in range(1, len(coor_list)):
            prev_coor = coor_list[i - 1]
            curr_coor = coor_list[i]
            
            # 计算两点之间的距离
            distance = haversine_distance(prev_coor[0], prev_coor[1], curr_coor[0], curr_coor[1])
            total_distance += distance
            total_pairs += 1
            
    return total_distance / total_pairs if total_pairs != 0 else 0

def remove_outliers_based_on_threshold(data_file, threshold):
    data = pd.read_csv(data_file)
    
    for index, row in data.iterrows():
        coor_list = eval(row['Coor_list'])
        node_list = eval(row['Node_list'])  # 读取Node列表
        
        filtered_coor_list = [coor_list[0]]  # 保留第一个点
        filtered_node_list = [node_list[0]]  # 保留第一个Node
        
        for i in range(1, len(coor_list)):
            prev_coor = filtered_coor_list[-1]
            curr_coor = coor_list[i]
            
            # 计算两点之间的距离
            distance = haversine_distance(prev_coor[0], prev_coor[1], curr_coor[0], curr_coor[1])
            
            # 如果距离小于阈值，则保留该点和相应的Node
            if distance < threshold:
                filtered_coor_list.append(curr_coor)
                filtered_node_list.append(node_list[i])
        
        data.at[index, 'Coor_list'] = str(filtered_coor_list)
        data.at[index, 'Node_list'] = str(filtered_node_list)  # 更新Node列表
    
    # 保存清理后的数据
    output_file_name = "visualize/cleaned_data.csv"
    data.to_csv(output_file_name, index=False)
    return output_file_name


# 计算前20条轨迹的平均距离
avg_distance = compute_avg_distance_for_first_20("data/rome/st_traj/matching_result.csv")
print(f"Average distance between points for first 20 trajectories: {avg_distance} km")

# 使用平均距离的2倍作为异常阈值
threshold = avg_distance * 4
print(f"Suggested threshold for outliers: {threshold} km")

# 使用该阈值来清除异常点
cleaned_file_name = remove_outliers_based_on_threshold("data/rome/st_traj/matching_result.csv", threshold)
print(f"Cleaned data saved to {cleaned_file_name}")

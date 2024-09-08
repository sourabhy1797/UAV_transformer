import pandas as pd
import folium
from visualize.eval import SimpleTester
import random
import torch
def read_nodes(file_path):
    df = pd.read_csv(file_path)
    node_coords = {row['node']: (row['lat'], row['lng']) for index, row in df.iterrows()}
    return node_coords

def visualize_nodes(node_coords, map, trajectory, col, node_sets=None):
    # Define a list of colors for different top-k node sets
    set_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

    # Add markers for each node in the trajectory
    for node in trajectory:
        coords = node_coords[node]
        folium.Marker(
            location=coords,
            popup=f'Node {node}',
            icon=folium.Icon(color=col),
        ).add_to(map)

    # Add lines between consecutive nodes in the trajectory
    for i in range(len(trajectory) - 1):
        folium.PolyLine(
            locations=[node_coords[trajectory[i]], node_coords[trajectory[i + 1]]],
            color=col
        ).add_to(map)

    # Visualize nodes from sets with different colors
    if node_sets:
        for index, node_set in enumerate(node_sets):
            current_color = set_colors[index % len(set_colors)]
            for node in node_set:
                coords = node_coords[node]
                folium.CircleMarker(
                    location=coords,
                    radius=8,  # Increase radius for better visibility
                    popup=f'Top-k Node {node}',
                    color=current_color,
                    fill=True,
                    fill_color=current_color
                ).add_to(map)

def main():
    torch.manual_seed(1953)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1953)
    random.seed(1953)
    node_file_path = 'data/rome/road/node.csv'
    model_path = 'model/rome/best.pkl'
    nodes = [3742,  4693,  4695,  4696,  4693,  4694, 20306,  3741, 20306,  4023,
          3745,  4023, 20302,  4022,  3744,  4022,  3735,  4022,  5434,  5433,
          5434,  4021,  5432,  4026,  4021,  4026,  7043, 29850, 16522, 29850,
         16522, 16519, 16520,  4141, 28559, 22945, 16513, 22945,  2271,  2269,
          2271,  2270,  4681,  2274,  2275,  2268,  2275,  2260,  2293,   165,
         26665,   165]

    node_coords = read_nodes(node_file_path)
    map_center = node_coords[nodes[0]]
    map = folium.Map(location=map_center, zoom_start=15)

    tester = SimpleTester(model_path=model_path)
    generated_trajectory, node_sets = tester.test_trajectory(nodes)

    visualize_nodes(node_coords, map, nodes, 'green')
    visualize_nodes(node_coords, map, generated_trajectory, 'blue', node_sets)

    map.save('visualize/map111.html')

if __name__ == '__main__':
    main()

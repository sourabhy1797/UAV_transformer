import sys
import osmnx as ox
import pandas as pd
import csv
import ray
import tqdm

ox.config(use_cache=True)  # Enable caching
G = ox.graph_from_place('San Francisco',network_type='all')
@ray.remote
def get_max_speed_osmnx(args):
    s_lng, s_lat, e_lng, e_lat = args
    # G = ox.graph_from_point((c_lat, c_lng), dist=1000, network_type='all')
    u = ox.distance.nearest_nodes(G, X=s_lng, Y=s_lat)
    v = ox.distance.nearest_nodes(G, X=e_lng, Y=e_lat)
    data = G.get_edge_data(u, v, 0)
    if data is None:
        nearest_edges = ox.distance.nearest_edges(G, X=e_lng, Y=e_lat)
        u, v, _ = nearest_edges
        data = G.get_edge_data(u, v, 0)
    maxspeed = data.get('maxspeed', None) #MPH
    length = data.get('length',None) # M
    if isinstance(maxspeed, list):   
        maxspeed = maxspeed[0]
    if maxspeed is None:
        maxspeed = 100
    return length, maxspeed

if __name__ == '__main__':

    ray.init()

    df = pd.read_csv('data/sanfrancisco/road/edge.csv')
    args_list = df[['s_lng', 's_lat', 'e_lng', 'e_lat']].values.tolist()

    # Using Ray to parallelize the computation
    futures = [get_max_speed_osmnx.remote(args) for args in args_list]

    # Setup tqdm for progress tracking
    progress_bar = tqdm.tqdm(total=len(futures), desc="Processing", position=0, leave=True)

    # Initialize results list
    results = []

    # Loop to fetch results and update progress bar
    while len(futures):
        done, futures = ray.wait(futures)
        results.extend(ray.get(done))
        progress_bar.update(len(done))

    progress_bar.close()

    print("Start")

    # Write results 
    with open('data/sanfrancisco/road/fake_edge.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['section_id','s_node','e_node','length','max_speed'])
        for i, result in enumerate(results):
            length, speed = result
            writer.writerow([i, df.loc[i,'s_node'], df.loc[i,'e_node'], length, speed])

    ray.shutdown()

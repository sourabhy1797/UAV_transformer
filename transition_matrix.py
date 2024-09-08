import pandas as pd
import pickle

data = pd.read_csv('data/rome/st_traj/cleaned_data.csv')

# Initialize a dictionary to save transition counts
transition_counts = {}

# Iterate through each trajectory in the dataset
for index, row in data.iterrows():
    node_list = eval(row['Node_list'])
    for i in range(len(node_list) - 1):
        current_node = node_list[i]
        next_node = node_list[i + 1]
        if current_node not in transition_counts:
            transition_counts[current_node] = {}
        if next_node not in transition_counts[current_node]:
            transition_counts[current_node][next_node] = 0
        transition_counts[current_node][next_node] += 1

# Initialize a new dictionary to save transition probabilities
transition_probabilities = {}

# Calculate transition probabilities
for current_node, transitions in transition_counts.items():
    total_transitions = sum(transitions.values())
    transition_probabilities[current_node] = [(node, count / total_transitions) for node, count in transitions.items()]

import pandas as pd

# 将transition_probabilities字典转换为适合DataFrame的格式
# 每一行包含一个当前节点、相邻节点和相应的转移概率
data_for_df = []
for current_node, transitions in transition_probabilities.items():
    for adjacent_node, probability in transitions:
        data_for_df.append({
            "Current_Node": current_node,
            "Adjacent_Node": adjacent_node,
            "Transition_Probability": probability
        })

# 创建DataFrame
df = pd.DataFrame(data_for_df)

# 保存为CSV文件
csv_filename = 'model/rome/transition_probabilities.csv'
df.to_csv(csv_filename, index=False)

print(f"Transition probabilities have been saved to {csv_filename}")

# Save as a dictionary format, each node corresponds to a list of adjacent points and probabilities
with open('model/rome/transition_probabilities_dict.pkl', 'wb') as file:
    pickle.dump(transition_probabilities, file)


# Load the transition probabilities dictionary from the file
with open('model/rome/transition_probabilities_dict.pkl', 'rb') as file:
    transition_probabilities = pickle.load(file)

# Printing everything could be too much, so here we print the transition probabilities for the first few nodes
# for node_id, transitions in list(transition_probabilities.items())[:5]:  # Print information for the first 5 nodes
#     print(f"Node {node_id}:")
#     for transition in transitions:
#         print(f"  to Node {transition[0]}: Probability {transition[1]}")
#     print()  # Empty line for separating outputs of different nodes


# Define the node_id you want to look up
node_id_to_lookup = 20

# Check if the node_id exists in the dictionary
if node_id_to_lookup in transition_probabilities:
    # Retrieve the transition probabilities for the specified node_id
    transitions = transition_probabilities[node_id_to_lookup]
    print(f"Transition probabilities for Node {node_id_to_lookup}:")
    for transition in transitions:
        print(f"  to Node {transition[0]}: Probability {transition[1]}")
else:
    print(f"No transition probabilities found for Node {node_id_to_lookup}")
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TrajectoryDataset(Dataset):
    def __init__(self, nodes_file, trajectories_file):
        # Load the node positions from nodes.csv
        self.nodes_df = pd.read_csv(nodes_file)
        self.node_positions = {row['node']: (row['centroid_x'], row['centroid_y'], row['centroid_z']) 
                               for _, row in self.nodes_df.iterrows()}
        # Dynamically calculate number of unique nodes
        self.num_nodes = len(self.nodes_df['node'].unique())

        # Load the trajectories from cleaned_data.csv
        self.trajectories_df = pd.read_csv(trajectories_file)
        self.trajectories = self.trajectories_df['Node_list'].apply(eval).tolist()

        # Dynamically calculate the max sequence length
        self.max_seq_len = max(len(trajectory) for trajectory in self.trajectories)

    def __len__(self):
        return len(self.trajectories)

    def pad_sequence(self, sequence):
        # Pad the sequence with (-1, -1, -1) to match max_seq_len
        padded_seq = sequence + [(-1, -1, -1)] * (self.max_seq_len - len(sequence))
        return padded_seq

    def __getitem__(self, idx):
        # Get trajectory by index and retrieve positions for nodes
        trajectory = self.trajectories[idx]
        positions = [self.node_positions[node_id] for node_id in trajectory if node_id in self.node_positions]

        # Pad the positions sequence
        padded_positions = self.pad_sequence(positions)

        # Generate the target, which is the next node in the sequence
        target = trajectory[1:] + [-1] * (self.max_seq_len - len(trajectory))  # Targets are padded with -1

        # Convert to tensor: input is positions, target is the next node sequence
        return torch.tensor(padded_positions, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

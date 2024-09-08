import torch
from model_network import TransformerGCN
import data_utils
import numpy as np
import random
import pandas as pd
import yaml
import pickle
import csv
import scipy.io
import matplotlib.pyplot as plt

class SimpleTester(object):
    def __init__(self):
        self.adjacency_dict = self.load_adjacency_set()  # Load adjacency set

    def load_adjacency_set(self, filepath='model/sanfrancisco/adjacency_set.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    def save_pois_to_csv(self, pois, filename):
        # 将POIs保存到CSV文件
        with open(filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['POI_ID'])  # 写入标题行
            for poi in pois.numpy():
                csv_writer.writerow([poi])  # 写入每个POI ID    
    def load_model(self, model_path='model/sanfrancisco/best.pkl'):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        model = TransformerGCN(
            feature_size=config["feature_size"],
            date2vec_size=config["date2vec_size"],
            dropout_rate=config["dropout_rate"],
            embedding_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        model.load_state_dict(torch.load(model_path))
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()
        self.road_network = data_utils.load_network(str(config["dataset"])).to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

    def load_node_info_and_pois(self):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.nodes = torch.tensor(pd.read_csv("./data/" + str(config["dataset"]) + "/road/node.csv", sep=',')[['lng','lat']].to_numpy(), dtype=torch.double).to(self.device)
        # Randomly select 100 POIs
        self.pois = torch.randint(0, 9860, (100,))
        self.save_pois_to_csv(self.pois, 'pois.csv')

        
        
    def test(self, node_sequence, top_k=10):
        road_network = data_utils.load_network(self.dataset).to(self.device)
        
        # Convert input node_id to a tensor and adjust dimensions for model input
        node_batch = torch.tensor([node_sequence], dtype=torch.int64).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(road_network, node_batch)
            prob_distribution = torch.softmax(outputs.squeeze(), dim=-1)
        
        # Convert to numpy array for easier processing
        prob_distribution = prob_distribution.cpu().numpy()
        
        # Get top-K nodes and their probabilities
        top_k_indices = np.argsort(prob_distribution)[::-1][:top_k]
        top_k_probs = prob_distribution[top_k_indices]
        
        return top_k_indices, top_k_probs
    
    def generate_trajectory(self, node_id):
        trajectory = node_id  # Initialize trajectory with start node

        input_sequence = torch.tensor([trajectory], dtype=torch.int64).to(self.device)
        with torch.no_grad():
            output_probs = self.model(self.road_network, input_sequence)
        next_traject = torch.argmax(output_probs,dim = 2).tolist()
        top_k_indices = np.argsort(output_probs.squeeze().cpu().numpy(), axis=1)[:, ::-1][:, :9860].tolist()
        # output_probs_np = torch.nn.Softmax(dim = -1)(output_probs).cpu().numpy()
        output_probs_np = output_probs.cpu().numpy()

        return next_traject[0], top_k_indices, output_probs_np
    
    # Fill the utility loss matrix
    def calculate_utility_loss_matrix(self, trajectory_example, mode='matrix'):
        """
        calculate_utility_loss_matrix
        :param file: 
        X: weight, K: the top k nodes you need
        """
        trajectory_length = len(trajectory_example)
        utility_loss_array = torch.zeros((trajectory_length, 9860), device=self.device, dtype=torch.double)
        # Start calculating from the second real_node
        for i in range(1, trajectory_length):
            previous_node = trajectory_example[i - 1]
            current_node = trajectory_example[i]
            # Determine fake_nodes in the location set of the current real_node
            previous_reachable_nodes = self.adjacency_dict[previous_node]
            for fake_node in previous_reachable_nodes:
                lon_real, lat_real = self.nodes[current_node, 0], self.nodes[current_node, 1]
                lon_fake, lat_fake = self.nodes[fake_node,0], self.nodes[fake_node,1]
                lon_poi, lat_poi = self.nodes[self.pois,0], self.nodes[self.pois,1]
                distance_real_poi = torch.mean(data_utils.haversine_distance(lon_real, lat_real, lon_poi, lat_poi) **2) * 1000
                distance_fake_poi = torch.mean(data_utils.haversine_distance(lon_fake, lat_fake, lon_poi, lat_poi) **2) * 1000
                if mode == 'score':
                    if abs(distance_real_poi - distance_fake_poi) > 0:
                        utility_loss_array[i][fake_node] = 1.0 / abs(distance_real_poi - distance_fake_poi)
                    # else:
                    #     utility_loss_array[i][fake_node] = 100000
                else:
                    utility_loss_array[i][fake_node] = abs(distance_real_poi - distance_fake_poi)
            # if mode == 'score':
            #     utility_loss_array[i] = torch.nn.Softmax(dim = -1)(utility_loss_array[i])
            # else:
            #     print(utility_loss_array.max(dim=-1))
        # print(mode + " mean utility loss: " + str(utility_loss_array.mean(dim=-1)))
        return utility_loss_array.cpu().numpy()
    
    def test_trajectory(self, node_id, max_length=50):
        trajectory_generate, node_sets, output_probs_np = self.generate_trajectory(node_id)

        # Create a zero matrix with the same shape as output_probs_np
        modified_probs_np = np.zeros_like(output_probs_np)
        modified_probs_np[0] = output_probs_np[0]
        
        # Filter the location set from the second point based on the adjacency set of the previous point
        for i in range(1, len(node_sequence)):
            previous_node_adjacency = self.adjacency_dict.get(node_sequence[i-1], set())
            for j in range(output_probs_np.shape[1]):
                if j in previous_node_adjacency:
                    modified_probs_np[i, j] = output_probs_np[i, j]
        return trajectory_generate, modified_probs_np, node_sets
    
    def check_trajectory(self, trajectory):
        adjacency_dict = self.load_adjacency_set()  # Ensure adjacency set is loaded
        validity_list = []
        for i in range(len(trajectory) - 1):
            current_node = trajectory[i]
            next_node = trajectory[i + 1]
            # Check if the next point is in the adjacency set of the current point
            if next_node in adjacency_dict.get(current_node, set()):
                validity_list.append(True)
            else:
                validity_list.append(False)
        return validity_list
        
    def save_scores_to_csv(self, scores, trajectory, X_value, filename_template='{}.csv'):
        filename = filename_template.format(X_value)  # Generate filename based on X_value
        with open(filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            # The location set and scores for the first node are not filtered
            print(trajectory, len(trajectory))
            real_node_id = trajectory[0]
            sorted_indices = np.argsort(scores[0])[::-1]
            filtered_indices = [real_node_id + 1, [idx+1 for idx in sorted_indices if idx in self.adjacency_dict.get(real_node_id, set())]]
            csv_writer.writerow(filtered_indices)
            for i in range(1, len(trajectory)):
                real_node_id = trajectory[i]
                # Get the set of reachable nodes for the previous real node
                reachable_nodes = self.adjacency_dict.get(trajectory[i - 1], set()) if i > 0 else set()
                # Sort scores in descending order
                sorted_indices = np.argsort(scores[i])[::-1]
                # Filter the location set, keeping only reachable nodes
                filtered_indices = [idx for idx in sorted_indices if idx in reachable_nodes]
                return_filtered_indices = [(idx + 1) for idx in filtered_indices]
                # Get the scores after filtering
                filtered_scores = [scores[i][idx] for idx in filtered_indices]

                # Write the current row data, including scores
                # csv_writer.writerow([real_node_id + 1, filtered_indices, filtered_scores])
                csv_writer.writerow([real_node_id + 1, return_filtered_indices])




if __name__ == "__main__":
    torch.manual_seed(1953)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1953)
    random.seed(1953)

    """
    Experiment when change weight and K
    """
   
    mat = list(scipy.io.loadmat('obf_trajs.mat')['obf_trajs'])
    tester = SimpleTester()
    tester.load_model()
    # node_sequence = [18429, 14796, 2924, 18179, 2923,2498, 21058, 7099, 2935, 2061, 2556,2557, 2490 ,7910,2491,2492 ,7875,8757 ,8740,2650,2675]

    cnt = 0
    tester.load_node_info_and_pois()

    weight = [0.01, 0.001, 0.0001, 0.00001 ]
    # K = [5, 10, 20, 30]
    for node_sequence in mat:
        sample = [node-1 for node in node_sequence[0][0]]
        generated_trajectory, modified_probs_np, node_sets = tester.test_trajectory(sample)

        utility = tester.calculate_utility_loss_matrix(sample)
        utility_score = tester.calculate_utility_loss_matrix(sample, mode = 'score')
        for weight_idx, X in enumerate(weight):

            scores_total = utility_score + X* modified_probs_np[0]
            tester.save_scores_to_csv(scores_total, sample, './output/location_set'+str(cnt)  +'Weight' + str(X))

        cnt +=1
        # ind = np.argsort(scores_total, axis = -1)
        # for k_idx, k_value in enumerate(K):
        #     # For each k_value, keep the weight
        #     sorted = 0
        #     for candidate in range(9860-1,9860-k_value-1, -1):
        #         x = [utility[i, ind[i, candidate]] for i in range(1,len(node_sequence[0][0]))]
        #         sorted += np.mean(x)
        #         # print(np.mean(x))
        #     sorted /= k_value
            # results[traj_idx,weight_idx,k_idx] = sorted
            # print(results[traj_idx,weight_idx,:])
        # For each trajectory, generate the scores_matrix
        # for i in range(len(K)):
        #     ax.plot(weight, [results[traj_idx,j,i] for j in range(len(weight))], label='K = '+str(K[i]))
        #     ax.set_xticks(weight)
        #     ax.set_xscale('log')
        # ax.legend()
        # plt.savefig('./output/trajectory' + str(traj_idx) + '.jpg')
    # print(results[0,:,:])
    # with open('./output/utilityLoss.csv', mode='w', newline='') as file:
    #     csv_writer = csv.writer(file)
    #     head = ['traj_idx'] + ['weight_' + str(X) +' K_' + str(k_value) for X in weight for k_value in K]
    #     csv_writer.writerow(head)
    #     for traj_idx in range(100):
    #         row = [traj_idx] + list(np.resize(results[traj_idx], (len(weight) * len(K),)))
    #         csv_writer.writerow(row)
    # for weight_idx, X in enumerate(weight):
    #     for k_idx, k_value in enumerate(K):
    #         with open('./output/utilityLoss_new_w{}_k{}.csv'.format(X,k_value), mode='w', newline='') as file:
    #             csv_writer = csv.writer(file)
    #             for traj_idx, node_sequence in enumerate(mat):
    #                 csv_writer.writerow([traj_idx, len(node_sequence[0][0]), results[traj_idx,weight_idx, k_idx]])

    """
    General Test, keep weight and K, generate location set for each trajectory
    """
    # mat = list(scipy.io.loadmat('new_traj.mat')['new_traj'])
    # tester = SimpleTester()
    # tester.load_model()

    # tester.load_node_info_and_pois()
    # cnt = 0
    # weight = 10
    # for node_sequence in mat:
    #     generated_trajectory, modified_probs_np, node_sets = tester.test_trajectory(node_sequence[0][0])
    #     utility = tester.calculate_utility_loss_matrix(node_sequence[0][0], mode = 'score')
    #     scores_total = weight* utility + modified_probs_np[0]
    #     tester.save_scores_to_csv(scores_total, node_sequence[0][0], './output/location_set' + str(cnt) + '.csv')
    #     cnt +=1

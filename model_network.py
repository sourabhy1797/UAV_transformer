import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GCN(nn.Module):
    def __init__(self, feature_size, embedding_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_size, embedding_size, cached=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x)
        # (num_nodes, embedding_size)
        return x
    
class TrajEmbedding(nn.Module):
    def __init__(self, feature_size, embedding_size, device):
        super(TrajEmbedding, self).__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.device = device
        self.gcn = GCN(feature_size, embedding_size).to(self.device)

    def forward(self, network, traj_seqs):
        """
        padding and spatial embedding trajectory with network topology
        :param network: the Pytorch geometric data object
        :param traj_seqs: list [batch,node_seq]
        :return: packed_input
        """
        batch_size = len(traj_seqs)
        ## Embedding decode
        seq_lengths = []
        for traj in traj_seqs:
            traj = list(traj)
            if -1 in traj:
                seq_lengths.append(traj.index(-1))
            else:
                seq_lengths.append(len(traj)) 

        # prepare sequence tensor
        embedded_seq_tensor = torch.zeros((batch_size, len(traj_seqs[0]), self.embedding_size), dtype=torch.float32)
        attention_mask = torch.zeros(batch_size, len(traj_seqs[0]),dtype=torch.bool)

        # get node embeddings from gcn
        # (num_nodes, embedding_size)
        node_embeddings = self.gcn(network)
        # node_embeddings_numpy = node_embeddings.detach().cpu().numpy()
        
        # # Save as CSV
        # df = pd.DataFrame(node_embeddings_numpy)
        # df.to_csv('data_smooth.csv', index=False)
        
        # print(node_embeddings)
        # node_embeddings = torch.tensor(np.load('data/rome/node_features.npy'), dtype=torch.float).to(self.device)
        # get embedding for trajectory embeddings
        for idx, (seq, seqlen) in enumerate(zip(traj_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = node_embeddings.index_select(0, seq[:seqlen].long())
            attention_mask[idx, :seqlen] = 1

        embedded_seq_tensor = embedded_seq_tensor.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths, batch_first=True, enforce_sorted=False)

        return embedded_seq_tensor, attention_mask
    
class TransformerGCN(nn.Module):
    def __init__(self, feature_size, date2vec_size, embedding_size, hidden_size, num_layers, dropout_rate, device='cuda: 0'):
        super(TransformerGCN, self).__init__()
        encoder_layer = TransformerEncoderLayer(embedding_size, 8, batch_first= True,dropout=dropout_rate)
        self.trajEmb = TrajEmbedding(feature_size, embedding_size, device)
        self.transformer = TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)
        self.embedding_size = embedding_size
        self.device = device
        self.fc = nn.Linear(embedding_size, hidden_size)

    def forward(self, network, traj_seqs):
        """
        :param network: the Pytorch geometric data object
        :param traj_seqs: list [batch,node_seq]
        :param time_seqs: list [batch,timestamp_seq]
        :return: the Spatio-Temporal embedding of trajectory
        """
        # Positional encodings
        traj_embeds, attention_mask = self.trajEmb(network, traj_seqs)
        traj_embeds += self.get_position_encodings(traj_embeds)  
        
        # Transformer
        # Transformer：传递掩码
        transf_outputs = self.transformer(traj_embeds, src_key_padding_mask=(~attention_mask).bool())
        # print(attention_mask.dtype)
        # Action prediction
        logits = self.fc(transf_outputs)
        return logits
    
    def get_position_encodings(self, seq):
        batch_size, max_len, d_model = seq.size()

        # 创建位置编码向量
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).repeat(batch_size, 1, 1).to(seq.device)  # 扩展到批次大小
        # print("Position Encodings:\n", pos_enc)

        return pos_enc
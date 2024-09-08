import torch
import torch.nn as nn

# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        # Move positional encoding to the same device as input x
        device = x.device
        return self.pe[:x.size(1), :].to(device)

# Transformer Encoder Model
class TrajectoryTransformer(nn.Module):
    def __init__(self, config, num_nodes, max_seq_len):
        super(TrajectoryTransformer, self).__init__()
        self.embedding = nn.Linear(config['model']['input_dim'], config['model']['embed_dim'])
        self.pos_encoder = PositionalEncoding(config['model']['embed_dim'], max_seq_len)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model']['embed_dim'], 
            nhead=config['model']['num_heads'],
            dropout=config['model']['dropout']
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['model']['num_layers'])
        
        self.fc = nn.Linear(config['model']['embed_dim'], num_nodes)  # Output layer now uses dynamic num_nodes

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoder(x)  # Ensure positional encoding is on the same device as x
        x = self.transformer_encoder(x)  # Using Transformer Encoder
        x = self.fc(x)
        return x

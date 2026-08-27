import torch
import torch.nn as nn
from ml.src.dataset import NUM_EMBEDDINGS, PAD_IDX

class NNUE(nn.Module):
    def __init__(self, accumulator_size=128):
        super().__init__()
        
        self.feature_embedding = nn.Embedding(
            num_embeddings=NUM_EMBEDDINGS,
            embedding_dim=accumulator_size,
            padding_idx=PAD_IDX
        )
        
        self.accumulator_bias = nn.Parameter(
            torch.zeros(accumulator_size)
        )
        
        self.output_network = nn.Sequential(
            nn.Linear(accumulator_size * 2, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def accumulate(self, features):
        embeddings = self.feature_embedding(features)
        
        accumulator = embeddings.sum(dim=1)
        
        accumulator = accumulator + self.accumulator_bias
        
        return torch.relu(accumulator) 
    
    def forward(self, white_features, black_features, side_to_move):
        white_accumulator = self.accumulate(white_features)
        black_accumulator = self.accumulate(black_features)
        
        stm = side_to_move.bool()
        
        first = torch.where(stm.unsqueeze(1), white_accumulator, black_accumulator)
        second = torch.where(stm.unsqueeze(1), black_accumulator, white_accumulator)
        
        x = torch.cat([first, second], dim=1)
        
        return self.output_network(x).squeeze(-1)
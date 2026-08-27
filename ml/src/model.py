import torch
import torch.nn as nn

def material_eval(board: torch.Tensor) -> torch.Tensor:
    """Return White's material advantage measured in pawns."""
    if board.ndim not in (3, 4):
        raise ValueError("Expected a board with shape (C, 8, 8) or (B, C, 8, 8)")
    if board.shape[-3] < 12:
        raise ValueError("Expected at least 12 piece channels")

    is_single_board = board.ndim == 3
    boards = board.unsqueeze(0) if is_single_board else board
    piece_values = boards.new_tensor([1, 3, 3, 5, 9, 0])
    piece_counts = boards[:, :12].sum(dim=(-2, -1))

    black_material = piece_counts[:, :6] @ piece_values
    white_material = piece_counts[:, 6:12] @ piece_values
    evaluation = (white_material - black_material).unsqueeze(-1)

    return evaluation[0] if is_single_board else evaluation

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return torch.relu(self.block(x) + x)
    
    
class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_network = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1),
            nn.ReLU(),
            
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.neural_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.conv_network(x)
        x = self.neural_network(x)
        return x
      
model = ChessCNN()      
     
loss_function = nn.HuberLoss(delta=1.0)

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


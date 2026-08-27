import torch
from torch.utils.data import Dataset, random_split, DataLoader 
import chess
from dataset import load_dataset

FRIENDLY_PIECES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4
}

ENEMY_PIECES = {
    chess.PAWN: 5,
    chess.KNIGHT: 6,
    chess.BISHOP: 7,
    chess.ROOK: 8,
    chess.QUEEN: 9,
    chess.KING: 10
}

NUM_PIECE_TYPES = 11
NUM_FEATURES = 64 * NUM_PIECE_TYPES * 64 # Total: 64*11*64 = 45,056
PAD_IDX = NUM_FEATURES
NUM_EMBEDDINGS = NUM_FEATURES + 1

def relative_piece_idx(piece, perspective):
    is_friendy = piece.color == perspective
    
    if is_friendy:
        return FRIENDLY_PIECES[piece.piece_type]
    else:
        return ENEMY_PIECES[piece.piece_type]

def orient_square(square, perspective):
    if perspective == chess.WHITE:
        return square
    
    return chess.square_mirror(square)

def feature_index(king_square, piece_index, piece_square):
    return (
        king_square * NUM_PIECE_TYPES * 64
        + piece_index * 64
        + piece_square
    )
    
def encode_position(board, perspective):
    king_square = orient_square(
        board.king(perspective),
        perspective
    )

    features = []

    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING and piece.color == perspective:
            continue

        piece_index = relative_piece_idx(piece, perspective)
        
        piece_square = orient_square(square, perspective)
        
        features.append(
            feature_index(
                king_square,
                piece_index,
                piece_square
            )
        )

    return features

MAX_EVAL = 10

class ChessNNUEDataset(Dataset):
    def __init__(self, positions):
        self.positions = [
            position 
            for position in positions
            if position["cp"] is not None   # We are not considering Mate positions
        ]
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, index):
        sample = self.positions[index]
        
        board = chess.Board(sample["fen"])
        
        white_features = encode_position(board, chess.WHITE)
        
        black_features = encode_position(board, chess.BLACK)
        
        side_to_move = board.turn
        
        evaluation = sample["cp"]
        
        target = max(-MAX_EVAL, min(evaluation/100, MAX_EVAL))
        
        if side_to_move == chess.BLACK:
            target *= -1
        
        return white_features, black_features, target, side_to_move
    
def pad_features(features_lists):
    max_length = max(
        len(features)
        for features in features_lists
    )
    
    batch_size = len(features_lists)
    
    padded = torch.full(
        (batch_size, max_length),
        PAD_IDX, 
        dtype = torch.long
    )
    
    for i, features in enumerate(features_lists):
        padded[i, :len(features)] = torch.tensor(
            features,
            dtype=torch.long
        )
        
    return padded

def nnue_collate_fn(batch):
    white_features, black_features, targets, stms = zip(*batch)
    
    white_features = pad_features(white_features)
    black_features = pad_features(black_features)
    
    targets = torch.tensor(targets, dtype=torch.float32)
    
    stms = torch.tensor(stms, dtype=torch.bool)
    
    return (
        white_features,
        black_features,
        targets,
        stms
    )
    
positions = load_dataset(
    "mateuszgrzyb/lichess-stockfish-normalized",
    split="train",
    streaming=True
)

NUM_POSITIONS = 250_000

positions = positions.take(NUM_POSITIONS)

positions = ChessNNUEDataset(list(positions))

train_len = int(len(positions) * 0.8)
val_len = int(len(positions) * 0.1)
test_len = int(len(positions) - train_len - val_len)

train_set, val_set, test_set = random_split(positions, [train_len, val_len, test_len])

torch.save(train_set, "ml/data/train_set.pt")
torch.save(val_set, "ml/data/val_set.pt")
torch.save(test_set, "ml/data/test_set.pt")
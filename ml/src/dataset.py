import torch
from torch.utils.data import Dataset as TorchDataset

import chess

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset

import os
import sys

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

class ChessNNUEDataset(TorchDataset):
    def __init__(self, positions):
        self.positions = positions
        
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
        
        # model expects target to be from stm perspective
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

def generate_positions(num_positions):
    positions = load_dataset(
            "mateuszgrzyb/lichess-stockfish-normalized",
            split=f"train",
            streaming=True
    )
    
    positions = positions.filter(
        lambda position: position["cp"] is not None
    )
    
    yield from positions.take(num_positions)

def main():
    num_positions = 50_000

    dataset = HFDataset.from_generator(
        generate_positions,
        gen_kwargs={"num_positions": num_positions},
        cache_dir="ml/data/cache",
        keep_in_memory=False
    )
    
    first_split = dataset.train_test_split(
        test_size=0.2,
        seed=10
    )
    
    second_split = first_split["test"].train_test_split(
        test_size=0.5,
        seed=10
    )
    
    splits = DatasetDict({
        "train": first_split["train"],
        "validation": second_split["train"],
        "test": second_split["test"]
    })
    
    splits.save_to_disk("ml/data/positions")
    
if __name__ == "__main__":
    main()
    
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Avoid PyArrow's deadlocked C++ thread-pool destructor
    os._exit(0)
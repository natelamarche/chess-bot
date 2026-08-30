import chess
import numpy as np
from datasets import DatasetDict, Features, Sequence, Value, load_from_disk
from dataset import MAX_EVAL, PAD_IDX, encode_position

MAX_POSITION_FEATURES = 31

ENCODED_FEATURES = Features({
    "white_features": Sequence(
        Value("uint16"),
        length=MAX_POSITION_FEATURES  
    ),
    "black_features": Sequence(
        Value("uint16"),
        length=MAX_POSITION_FEATURES  
    ),
    "target": Value("float32"),
    "side_to_move": Value("bool")
})

def encode_batch(batch):
    rows = []
    
    for fen, cp in zip(batch["fen"], batch["cp"]):
        board = chess.Board(fen)
        
        if not board.is_valid():
            continue
        
        white_indices = encode_position(board, chess.WHITE)
        black_indices = encode_position(board, chess.BLACK)

        target = np.clip(cp / 100.0, -MAX_EVAL, MAX_EVAL)
                
        if board.turn == chess.BLACK:
            target = -target

        rows.append(
            (white_indices, black_indices, target, board.turn)
        )

    size = len(rows)
    
    white = np.full(
        (size, MAX_POSITION_FEATURES),
        PAD_IDX,
        dtype=np.uint16
    )
    
    black = np.full(
        (size, MAX_POSITION_FEATURES),
        PAD_IDX,
        dtype=np.uint16
    )
    
    targets = np.empty(size, dtype=np.float32)
    
    sides_to_move = np.empty(size, dtype=np.bool_)
    
    for index, (white_indices, black_indices, target, side) in enumerate(rows):
        white[index, :len(white_indices)] = white_indices
        black[index, :len(black_indices)] = black_indices
        targets[index] = target
        sides_to_move[index] = side
        
    return {
        "white_features": white,
        "black_features": black,
        "target": targets,
        "side_to_move": sides_to_move
    }
        
def main():
    splits = load_from_disk("ml/data/positions")
    
    encoded_splits = splits.map(
        encode_batch,
        batched=True,
        batch_size=2_048,
        num_proc=24,
        remove_columns=splits["train"].column_names,
        features=ENCODED_FEATURES,
        desc="Encoding NNUE features"
    )
    
    encoded_splits.save_to_disk(
        "ml/data/encoded_positions"
    )
    
    print(f"Train len: {len(encoded_splits["train"])}")
    print(f"Validation len: {len(encoded_splits["validation"])}")
    print(f"Test len: {len(encoded_splits["test"])}")
    
    print("DONE")
    
if __name__ == "__main__":
    main()

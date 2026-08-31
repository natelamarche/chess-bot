import struct

import chess
import numpy as np
import torch

from ml.src.dataset import encode_position
from ml.src.model import NNUE
from ml.src.export_weights import MAGIC

WEIGHTS_PATH = "ml/model/model.nnue"
CHECKPOINT_PATH = "ml/model/model_weights.pth"

def read_array(file, shape):
    count = int(np.prod(shape))
    data = np.fromfile(file, dtype="<f4", count=count)
    
    if data.size != count:
        raise ValueError(
            f"Unexpected end of file: wanted {count} floats, "
            f"received {data.size}"
        )
    return data.reshape(shape)

def load_export(path):
    with open(path, "rb") as file:
        magic = file.read(8)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}")
        (
            format_version,
            feature_schema_version,
            num_features,
            accumulator_size,
            hidden1_size,
            hidden2_size,
            output_size
        ) = struct.unpack("<7I", file.read(28))
        
        weights = {
            "embedding": read_array(
                file, (num_features, accumulator_size)
            ),
            "accumulator_bias": read_array(
                file, (accumulator_size,)
            ),
            "layer1_weight": read_array(
                file, (hidden1_size, accumulator_size * 2)
            ),
            "layer1_bias": read_array(
                file, (hidden1_size,)
            ),
            "layer2_weight": read_array(
                file, (hidden2_size, hidden1_size)
            ),
            "layer2_bias": read_array(
                file, (hidden2_size,)
            ),
            "output_weight": read_array(
                file, (output_size, hidden2_size)
            ),
            "output_bias": read_array(
                file, (output_size,)
            )
        }
        if file.read(1):
            raise ValueError("Unexpected trailing data")
        
    print(
        f"Loaded format={format_version}, "
        f"schema={feature_schema_version}"
    )
    
    return weights

def relu(values):
    return np.maximum(values, 0.0)

def numpy_evaluate(board, weights):
    white_features = encode_position(board, chess.WHITE)
    black_features = encode_position(board, chess.BLACK)
    
    white_accumulator = (
        weights["accumulator_bias"]
        + weights["embedding"][white_features].sum(axis=0)
    )

    black_accumulator = (
        weights["accumulator_bias"]
        + weights["embedding"][black_features].sum(axis=0)
    )
    
    white_accumulator = relu(white_accumulator)
    black_accumulator = relu(black_accumulator)
    
    if board.turn == chess.WHITE:
        inputs = np.concatenate(
            [white_accumulator, black_accumulator]
        )
    else:
        inputs = np.concatenate(
            [black_accumulator, white_accumulator]
        )
    
    hidden1 = relu(
        weights["layer1_weight"] @ inputs
        + weights["layer1_bias"]
    )

    hidden2 = relu(
        weights["layer2_weight"] @ hidden1
        + weights["layer2_bias"]
    )

    output = (
        weights["output_weight"] @ hidden2
        + weights["output_bias"]
    )
    
    return float(output[0])

def pytorch_evaluate(board, model):
    white = torch.tensor(
        [encode_position(board, chess.WHITE)],
        dtype=torch.long,
    )
    black = torch.tensor(
        [encode_position(board, chess.BLACK)],
        dtype=torch.long,
    )
    side_to_move = torch.tensor(
        [board.turn],
        dtype=torch.bool,
    )

    with torch.no_grad():
        return model(white, black, side_to_move).item()
    
def main():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    
    model = NNUE(
        accumulator_size=checkpoint["accumulator_size"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    exported = load_export(WEIGHTS_PATH)

    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/5pk1/6p1/3P4/4P3/5K2/8/8 w - - 0 40",
        "r3k2r/ppp2ppp/2n5/3qp3/8/2N2N2/PPP2PPP/R2Q1RK1 b kq - 1 12",
    ]
    
    for fen in fens:
        board = chess.Board(fen)
        
        torch_result = pytorch_evaluate(board, model)
        export_result = numpy_evaluate(board, exported)
        
        difference = abs(torch_result - export_result)
        
        print(f"FEN:        {fen}")
        print(f"PyTorch:    {torch_result:.8f}")
        print(f"Export:     {export_result:.8f}")
        print(f"Difference: {difference:.10f}\n")

        if difference > 1e-5:
            raise AssertionError("Export validation failed")
    
    print("All exported evaluations match Pytorch")
    
    
if __name__ == "__main__":
    main()
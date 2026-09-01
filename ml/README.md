# NNUE training pipeline

This directory contains the end-to-end pipeline for the engine's learned evaluation: download labelled chess positions, encode sparse king-relative features, train the PyTorch model, evaluate it, and export weights for dependency-free C++ inference.

Run all commands from the repository root.

## Model

For each position, the encoder builds one feature list from White's perspective and one from Black's. A feature combines:

- The perspective king's square
- One of 11 relative piece types: five friendly non-king pieces and six enemy pieces
- The piece's square, vertically mirrored for Black

This produces `64 × 11 × 64 = 45,056` possible features. The perspective's own king defines the feature bucket and is not encoded as a piece; the opposing king is.

The network is intentionally small and incrementally updateable:

```text
active feature embeddings → sum + bias → ReLU  (128 values per perspective)
side-to-move accumulator first → concatenate   (256 values)
Linear 256→64 → ReLU → Linear 64→32 → ReLU → Linear 32→1
```

Targets are Stockfish centipawn evaluations converted to pawns, clipped to `[-10, 10]`, and expressed from the side-to-move perspective. Training uses Huber loss and Adam with early stopping.

The NNUE model currently shipped with the engine was trained on 50 million positions using a cloud-hosted NVIDIA GeForce RTX 5090.

## Environment

Python 3 with PyTorch is required. A CUDA device is used automatically when available.

```sh
python3 -m venv ml/.venv
source ml/.venv/bin/activate
python -m pip install -r ml/requirements.txt
```

## Pipeline

### 1. Download and split positions

```sh
python -m ml.src.dataset
```

Positions come from the Hugging Face dataset `mateuszgrzyb/lichess-stockfish-normalized`. The script filters missing evaluations and creates deterministic 80/10/10 train, validation, and test splits under `ml/data/positions`. Adjust `num_positions` in `dataset.py` before running to choose the dataset size.

### 2. Pre-encode sparse features

```sh
python ml/src/encode_dataset.py
```

This converts FEN strings into fixed-width `uint16` feature arrays under `ml/data/encoded_positions`, avoiding repeated board parsing during training. Tune `num_proc` in the script for the available CPU and memory.

### 3. Train

```sh
python -m ml.src.train
```

The best validation checkpoint is written to `ml/model/model_weights.pth`. The checkpoint includes the model state, optimizer state, accumulator size, feature-schema version, evaluation range, and validation loss.

### 4. Evaluate

```sh
python -m ml.src.evaluate
```

Evaluation reports MAE, MSE, sign accuracy, close-position sign accuracy, and MAE broken down by evaluation magnitude.

### 5. Export for C++

```sh
python -m ml.src.export_weights
```

The exporter writes `ml/model/model.nnue`, a versioned little-endian float32 binary containing the embeddings, accumulator bias, and dense-layer parameters. The padding embedding used for training batches is omitted.

Validate the exported model independently with NumPy:

```sh
python -m ml.test.test_export
```

The test loads the PyTorch checkpoint and exported binary separately, evaluates several FENs through both paths, and verifies numerical agreement. The C++ test suite then checks the same golden evaluations as well as incremental accumulator parity across quiet moves, captures, king moves, castling, en passant, and promotions.

## Artifacts

| File | Purpose |
|---|---|
| `ml/model/model_weights.pth` | Training checkpoint; required to resume, inspect, or re-export |
| `ml/model/model.nnue` | Runtime model loaded by the C++ engine |

The `.nnue` file is the deployable artifact. It is approximately 22 MB in the current float32 format; quantization is a possible future optimization.

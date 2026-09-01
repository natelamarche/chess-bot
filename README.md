# chess-bot

A chess engine written in C++20 with a learned, efficiently updatable neural evaluation. The engine combines legal move generation and alpha-beta search with an NNUE-style network trained in PyTorch, exported to a custom binary format, and evaluated directly in C++ without a machine-learning runtime.

## Highlights

- Complete legal move generation, including castling, en passant, and promotion
- Negamax search with alpha-beta pruning and mate/stalemate detection
- King-relative neural features evaluated from both players' perspectives
- Incremental feature accumulators: ordinary moves update only the embeddings that changed
- Custom, versioned `.nnue` model format with Python-to-C++ parity tests
- UCI interface for fixed-depth play in compatible GUIs and match runners
- Perft, feature-encoding, accumulator, inference, and search benchmarks

## How the evaluation works

Each active piece is encoded relative to one king, producing separate White and Black accumulators. At the root, an accumulator is the learned bias plus the embeddings of the active features. During search, quiet moves, captures, promotions, en passant, and castling update those sums incrementally. A move by the perspective king rebuilds only the affected accumulator because its king-relative feature bucket changes.

The two 128-value accumulators feed a small `256 → 64 → 32 → 1` network. Its result is a side-to-move score in pawns, converted to centipawns for search and UCI output. See [ml/README.md](ml/README.md) for the training and export pipeline.

The network currently bundled with the engine was trained in the cloud on 50 million labelled positions using an NVIDIA GeForce RTX 5090.

## Build

Requirements are CMake 3.16+ and a C++20 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

For the fastest floating-point inference, allow the compiler to reassociate and vectorize the dense reductions:

```sh
cmake -S . -B build-optimized \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -ffast-math"
cmake --build build-optimized --parallel
```

On Apple Silicon, add `-DCMAKE_OSX_ARCHITECTURES=arm64` if your CMake installation otherwise produces an x86-64 binary.

The engine currently loads `ml/model/model.nnue` relative to its working directory, so run it from the repository root.

## Use

Launch the terminal game:

```sh
./build/chess_cli
```

Exercise the UCI engine directly:

```sh
printf 'uci\nisready\nposition startpos moves e2e4 e7e5\ngo depth 5\nquit\n' \
  | ./build/engine
```

Supported commands include `uci`, `isready`, `ucinewgame`, `position startpos`, `position fen`, move lists, `go depth`, and `quit`. Search defaults to depth 5 when no depth is supplied. Time management and asynchronous `stop` are intentionally outside the current scope.

## Test and benchmark

```sh
ctest --test-dir build --output-on-failure
./build/perft_test 4
./build/search_bench 4
```

The v1.3.0 release reached 760,881 nodes per second at depth 6 in the recorded local benchmark. Incremental accumulators improved throughput by about 24% over v1.2.0, which rebuilt both accumulators at every leaf. Results are hardware-dependent; complete historical output lives under [`releases/`](releases/).

## Project layout

| Path | Purpose |
|---|---|
| `include/chess`, `src/chess` | Board representation, moves, and legal move generation |
| `include/engine`, `src/engine` | Search, NNUE features, state, model loading, and inference |
| `src/uci` | UCI command loop |
| `ml` | Dataset preparation, PyTorch training, evaluation, and export |
| `tests` | Perft, NNUE parity, and search benchmarks |
| `releases` | Versioned benchmark records |

## License

Released under the [MIT License](LICENSE).

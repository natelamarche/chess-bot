# chess-bot

C++ chess engine with a command-line game and a UCI executable.

## Build

```sh
cmake -S . -B build
cmake --build build
```

Run `build/chess_uci` from a UCI-compatible chess GUI, or exercise it directly:

```text
uci
isready
position startpos moves e2e4 e7e5
go depth 5
quit
```

The UCI executable supports `uci`, `isready`, `ucinewgame`, `position startpos`,
`position fen`, move lists, `go depth`, and `quit`. When no depth is supplied,
the engine searches to depth 5.

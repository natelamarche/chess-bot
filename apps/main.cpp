#include <iostream>
#include "chess/board.h"
#include "perft/perft.h"

int main() {
    chess::Board b;
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::cout << b.to_string() << "\n";

    for (int depth = 0; depth < 5; depth++){
        std::cout << "perft(" << depth << ") = " << chess::perft(b, depth) << "\n";
    }

    b.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    std::cout << b.to_string() << "\n";

    for (int depth = 0; depth < 5; depth++){
        std::cout << "perft(" << depth << ") = " << chess::perft(b, depth) << "\n";
    }

    return 0;
}

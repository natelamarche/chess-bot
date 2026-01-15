#include <iostream>
#include "chess/board.h"

int main() {
    chess::Board b;
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::cout << b.to_string() << "\n";

    b.set_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    std::cout << b.to_string() << "\n";

    return 0;
}

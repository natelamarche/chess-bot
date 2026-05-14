#include "perft/perft.h"
#include "chess/movegen.h"
#include <vector>

namespace chess {
    
std::uint64_t perft(const Board& board, int depth){

    if (depth == 0){
        return 1;
    }

    std::vector<Move> moveset = generate_legal_moves(board);

    std::uint64_t nodes = 0;
    for (const Move& move : moveset){
        Board next = board;
        next.make_move(move);

        nodes += perft(next, depth - 1);

    }

    return nodes;

}

}

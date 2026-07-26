#include "perft/perft.h"
#include "chess/movegen.h"
#include <vector>

namespace chess {
    
std::uint64_t perft(Board& board, int depth){

    if (depth == 0){
        return 1;
    }

    std::vector<Move> moveset = generate_legal_moves(board);

    std::uint64_t nodes = 0;
    for (const Move& move : moveset){
        UndoState undo_state = board.make_move(move);
    
        nodes += perft(board, depth - 1);

        board.unmake_move(move, undo_state);

    }

    return nodes;

}

}

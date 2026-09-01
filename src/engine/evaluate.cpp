#include "engine/evaluate.h"
#include "engine/nnue_state.h"
#include <array>
#include <cmath>

namespace chess::engine {

int evaluate(const Board& board, const NnueState& state, const NnueModel& model){

    const float pawns = model.forward(
        state.white(), 
        state.black(), 
        board.side_to_move()
    );

    return static_cast<int>(std::lround(pawns * 100.0F));

}

}
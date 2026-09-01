#include "engine/evaluate.h"
#include "engine/nnue_state.h"
#include <array>
#include <cmath>

namespace chess::engine {

int evaluate(const Board& board, const NnueModel& model){

    const auto white = nnue::rebuild_accumulator(board, Colour::White, model);
    const auto black = nnue::rebuild_accumulator(board, Colour::Black, model);

    const float pawns = model.forward(white, black, board.side_to_move());

    return static_cast<int>(std::lround(pawns * 100.0F));

}

}
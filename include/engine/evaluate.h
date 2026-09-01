#pragma once
#include "chess/board.h"
#include "engine/nnue_model.h"
#include "engine/nnue_state.h"

namespace chess::engine {

int evaluate(const Board& board, const NnueState& state, const NnueModel& model);

}
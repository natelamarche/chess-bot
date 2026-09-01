#pragma once
#include "chess/board.h"
#include "engine/nnue_model.h"

namespace chess::engine {

int evaluate(const Board& board, const NnueModel& model);

}
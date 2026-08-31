#pragma once

#include <vector>

#include "chess/board.h"
#include "engine/nnue_model.h"

namespace chess::engine::nnue {

using Accumulator = std::vector<float>;

Accumulator rebuild_accumulator(const Board& board, Colour perspective, const NnueModel& model);

} // namespace chess::engine::nnue
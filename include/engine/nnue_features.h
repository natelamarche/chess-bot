#pragma once

#include <vector>
#include <cstdint>
#include "chess/board.h"

namespace chess::engine::nnue {

std::uint32_t feature_index(Square king_square, std::uint32_t piece_index, Square piece_square);

std::vector<std::uint32_t> active_features(const Board& board, Colour perspective);

} // namespace chess::engine::nnue

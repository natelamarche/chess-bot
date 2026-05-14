#pragma once
#include <cstdint>
#include "chess/board.h"

namespace chess {

std::uint64_t perft(const Board& board, int depth);

}


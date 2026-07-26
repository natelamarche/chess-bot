#pragma once
#include <cstdint>
#include "chess/board.h"

namespace chess {

std::uint64_t perft(Board& board, int depth);

}


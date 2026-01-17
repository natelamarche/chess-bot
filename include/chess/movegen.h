#pragma once
#include <vector>
#include "chess/board.h"
#include "chess/move.h"


namespace chess {

std::vector<Move> generate_pawn_moves(const Board& board);

}
#pragma once
#include <vector>
#include "chess/board.h"
#include "chess/move.h"


namespace chess {

std::vector<Move> generate_pseudo_legal_moves(const Board& board);

std::vector<Move> generate_legal_moves(const Board& board);

}

#pragma once
#include <vector>
#include "chess/board.h"
#include "chess/move.h"


namespace chess {

bool in_check(const Board& board, Colour clr);

std::vector<Move> generate_pseudo_legal_moves(const Board& board);

std::vector<Move> generate_legal_moves(const Board& board);

}

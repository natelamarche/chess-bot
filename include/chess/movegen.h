#pragma once
#include <vector>
#include <array>
#include "chess/board.h"
#include "chess/move.h"


namespace chess {

std::vector<Move> generate_pawn_moves(const Board& board);

std::vector<Move> generate_knight_moves(const Board& board);

std::vector<Move> generate_rook_moves(const Board& board);

std::vector<Move> generate_bishop_moves(const Board& board);

std::vector<Move> generate_queen_moves(const Board& board);

std::vector<Move> generate_king_moves(const Board& board);

}

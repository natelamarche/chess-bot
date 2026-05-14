#pragma once
#include "chess/board.h"
#include "chess/move.h"

namespace chess::engine {

struct SearchResult {
    Move move{};
    int score{};
};

int negamax(const Board& board, int depth, int alpha, int beta);

SearchResult search_best_move(const Board& board, int depth);

}


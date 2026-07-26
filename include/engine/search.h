#pragma once

#include "chess/board.h"
#include "chess/move.h"

namespace chess::engine {

struct SearchResult {
        Move move{};
        int score{};
    };

struct SearchStats {
    std::uint64_t nodes = 0;
    std::uint64_t quiescence_nodes = 0;
    std::uint64_t beta_cutoffs = 0;

};

class Searcher {
public:
    SearchResult search_best_move(const Board& board, int depth);

    const SearchStats& getStats() const {
        return stats_;
    }

    void resetStats();


private:
    int negamax(const Board& board, int depth, int alpha, int beta);

    SearchStats stats_;


};


}


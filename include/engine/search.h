#pragma once

#include "chess/board.h"
#include "chess/move.h"
#include "engine/nnue_model.h"

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
    explicit Searcher(const NnueModel& model) : model_{model} {};

    SearchResult search_best_move(Board& board, int depth);

    const SearchStats& getStats() const {
        return stats_;
    }

    void resetStats();


private:
    const NnueModel& model_;

    int negamax(Board& board, int depth, int alpha, int beta);

    SearchStats stats_;


};


}


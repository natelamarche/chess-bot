#pragma once

#include <array>

#include "chess/board.h"
#include "chess/move.h"
#include "engine/nnue_model.h"

namespace chess::engine {

static constexpr std::size_t accumulator_size = 128;
using Accumulator = std::array<float, accumulator_size>;


struct PieceChange {
    PieceOnSquare piece;
    Square square;
};

struct MoveFeatureChanges {
    std::array<PieceChange, 2> removed{};
    std::array<PieceChange, 2> added{};

    std::size_t removed_count{};
    std::size_t added_count{};

    bool king_moved{};
    Colour king_colour{Colour::White};

    void remove(PieceOnSquare piece, Square square) {
        removed[removed_count++] = {piece, square};    
    }

    void add(PieceOnSquare piece, Square square) {
        added[added_count++] = {piece, square};
    }
};

MoveFeatureChanges prepare_move(const Board& board, const Move& move);

class NnueState {

public:

    void initialize(const Board& board, const NnueModel& model);

    void apply_move(const MoveFeatureChanges& changes, const Board& board_after, const NnueModel& model);

    const Accumulator& white() const noexcept {
        return white_;
    }
    
    const Accumulator& black() const noexcept {
        return black_;
    }

private:
    Accumulator white_{};
    Accumulator black_{};
    bool initialized_{false};
};

} // namespace chess::engine

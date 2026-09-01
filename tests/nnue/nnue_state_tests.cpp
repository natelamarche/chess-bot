#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>
#include <string>

#include "chess/board.h"
#include "chess/movegen.h"
#include "engine/nnue_model.h"
#include "engine/nnue_state.h"

namespace {

bool accumulators_match(
    std::span<const float> actual,
    std::span<const float> expected,
    const std::string& description
) {
    if (actual.size() != expected.size()) {
        std::cerr << description << ": accumulator sizes differ\n";
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-4F) {
            std::cerr << description << ": mismatch at index " << i
                      << " (actual " << actual[i]
                      << ", expected " << expected[i] << ")\n";
            return false;
        }
    }

    return true;
}

bool expect_incremental_move(
    const chess::engine::NnueModel& model,
    const std::string& description,
    const std::string& fen,
    int from,
    int to,
    chess::Piece promotion = chess::Piece::None
) {
    chess::Board board;
    board.set_fen(fen);

    const auto legal_moves = chess::generate_legal_moves(board);
    const auto match = std::find_if(
        legal_moves.begin(),
        legal_moves.end(),
        [from, to, promotion](const chess::Move& move) {
            return move.from == from
                && move.to == to
                && move.promotion == promotion;
        }
    );

    if (match == legal_moves.end()) {
        std::cerr << description << ": test move is not legal\n";
        return false;
    }

    chess::engine::NnueState incremental;
    incremental.initialize(board, model);

    const chess::engine::MoveFeatureChanges changes =
        chess::engine::prepare_move(board, *match);

    board.make_move(*match);
    incremental.apply_move(changes, board, model);

    chess::engine::NnueState rebuilt;
    rebuilt.initialize(board, model);

    return accumulators_match(
               incremental.white(),
               rebuilt.white(),
               description + " white"
           )
        && accumulators_match(
               incremental.black(),
               rebuilt.black(),
               description + " black"
           );
}

} // namespace

int main() {
    chess::engine::NnueModel model;
    model.load("ml/model/model.nnue");

    if (!expect_incremental_move(
            model,
            "quiet move",
            "rnbqkbnr/pppppppp/8/8/8/8/"
            "PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            12,
            28
        )) {
        return 1;
    }

    if (!expect_incremental_move(
            model,
            "capture",
            "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
            28,
            35
        )) {
        return 1;
    }

    if (!expect_incremental_move(
            model,
            "king move",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            4,
            12
        )) {
        return 1;
    }

    if (!expect_incremental_move(
            model,
            "white king-side castle",
            "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
            4,
            6
        )) {
        return 1;
    }

    if (!expect_incremental_move(
            model,
            "black queen-side castle",
            "r3k3/8/8/8/8/8/8/4K3 b q - 0 1",
            60,
            58
        )) {
        return 1;
    }

    if (!expect_incremental_move(
            model,
            "en passant",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            36,
            43
        )) {
        return 1;
    }

    if (!expect_incremental_move(
            model,
            "promotion",
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            48,
            56,
            chess::Piece::Queen
        )) {
        return 1;
    }

    if (!expect_incremental_move(
            model,
            "capture promotion",
            "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            48,
            57,
            chess::Piece::Queen
        )) {
        return 1;
    }

    std::cout << "NNUE state tests passed\n";
    return 0;
}

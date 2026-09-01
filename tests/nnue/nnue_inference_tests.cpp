#include <cmath>
#include <iostream>
#include <string>

#include "chess/board.h"
#include "engine/nnue_model.h"
#include "engine/nnue_state.h"

namespace {

bool expect_evaluation(
    const chess::engine::NnueModel& model,
    const std::string& fen,
    float expected
) {
    chess::Board board;
    board.set_fen(fen);

    const auto white = chess::engine::nnue::rebuild_accumulator(
        board,
        chess::Colour::White,
        model
    );
    const auto black = chess::engine::nnue::rebuild_accumulator(
        board,
        chess::Colour::Black,
        model
    );

    const float actual = model.forward(
        white,
        black,
        board.side_to_move()
    );

    if (std::abs(actual - expected) <= 1e-4F) {
        return true;
    }

    std::cerr << "Evaluation mismatch\n"
              << "FEN: " << fen << '\n'
              << "Expected: " << expected << '\n'
              << "Actual: " << actual << '\n';
    return false;
}

} // namespace

int main() {
    chess::engine::NnueModel model;
    model.load("ml/model/model.nnue");

    if (!expect_evaluation(
            model,
            "rnbqkbnr/pppppppp/8/8/8/8/"
            "PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            0.22840843F
        )) {
        return 1;
    }

    if (!expect_evaluation(
            model,
            "8/5pk1/6p1/3P4/4P3/5K2/8/8 w - - 0 40",
            0.31382689F
        )) {
        return 1;
    }

    if (!expect_evaluation(
            model,
            "r3k2r/ppp2ppp/2n5/3qp3/8/2N2N2/"
            "PPP2PPP/R2Q1RK1 b kq - 1 12",
            -3.28155994F
        )) {
        return 1;
    }

    std::cout << "NNUE inference tests passed\n";
    return 0;
}

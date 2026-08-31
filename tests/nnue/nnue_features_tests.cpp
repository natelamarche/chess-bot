#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "chess/board.h"
#include "engine/nnue_features.h"

namespace {

bool expect_features(
    const chess::Board& board,
    chess::Colour perspective,
    std::vector<std::uint32_t> expected
) {
    std::vector<std::uint32_t> actual =
        chess::engine::nnue::active_features(board, perspective);

    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());

    if (actual == expected) {
        return true;
    }

    std::cerr << "Feature mismatch for "
              << (perspective == chess::Colour::White ? "White" : "Black")
              << " perspective\nExpected:";
    for (std::uint32_t feature : expected) {
        std::cerr << ' ' << feature;
    }

    std::cerr << "\nActual:";
    for (std::uint32_t feature : actual) {
        std::cerr << ' ' << feature;
    }
    std::cerr << '\n';

    return false;
}

} // namespace

int main() {
    chess::Board board;
    board.set_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");

    if (!expect_features(
            board,
            chess::Colour::White,
            {2844, 3427, 3516}
        )) {
        return 1;
    }

    if (!expect_features(
            board,
            chess::Colour::Black,
            {3099, 3172, 3516}
        )) {
        return 1;
    }

    std::cout << "NNUE feature tests passed\n";
    return 0;
}

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "chess/board.h"
#include "engine/nnue_model.h"
#include "engine/nnue_state.h"

namespace {

bool accumulators_match(
    const chess::engine::nnue::Accumulator& actual,
    const std::vector<float>& expected
) {
    if (actual.size() != expected.size()) {
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-5F) {
            return false;
        }
    }

    return true;
}

std::vector<float> expected_accumulator(
    const chess::engine::NnueModel& model,
    const std::vector<std::uint32_t>& features
) {
    const auto bias = model.accumulator_bias();
    std::vector<float> expected(bias.begin(), bias.end());

    for (std::uint32_t feature : features) {
        const float* weights = model.feature_weights(feature);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            expected[i] += weights[i];
        }
    }

    return expected;
}

} // namespace

int main() {
    chess::Board board;
    board.set_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");

    chess::engine::NnueModel unloaded_model;
    try {
        chess::engine::nnue::rebuild_accumulator(
            board,
            chess::Colour::White,
            unloaded_model
        );
        std::cerr << "Expected an unloaded model to be rejected\n";
        return 1;
    } catch (const std::logic_error&) {
    }

    chess::engine::NnueModel model;
    model.load("ml/model/model.nnue");

    const auto white = chess::engine::nnue::rebuild_accumulator(
        board,
        chess::Colour::White,
        model
    );
    const auto expected_white = expected_accumulator(
        model,
        {2844, 3427, 3516}
    );

    if (!accumulators_match(white, expected_white)) {
        std::cerr << "White accumulator did not match expected values\n";
        return 1;
    }

    const auto black = chess::engine::nnue::rebuild_accumulator(
        board,
        chess::Colour::Black,
        model
    );
    const auto expected_black = expected_accumulator(
        model,
        {3099, 3172, 3516}
    );

    if (!accumulators_match(black, expected_black)) {
        std::cerr << "Black accumulator did not match expected values\n";
        return 1;
    }

    std::cout << "NNUE state tests passed\n";
    return 0;
}

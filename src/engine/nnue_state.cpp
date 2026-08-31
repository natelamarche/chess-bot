#include "engine/nnue_state.h"
#include <stdexcept>
#include "engine/nnue_features.h"

namespace chess::engine::nnue {

Accumulator rebuild_accumulator(const Board& board, Colour perspective, const NnueModel& model) {
    if (!model.loaded()) {
        throw std::logic_error("NNUE model is not loaded");
    }

    const auto bias = model.accumulator_bias();

    Accumulator accumulator(bias.begin(), bias.end());

    for (std::uint32_t feature : active_features(board, perspective)){
        const float* weights = model.feature_weights(feature);

        for (std::size_t i = 0; i < accumulator.size(); i++) {
            accumulator[i] += weights[i];
        }

    }

    return accumulator;

}

} // namespace chess::engine::nnue
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>
#include "chess/types.h"

namespace chess::engine {

class NnueModel {
public:
    static constexpr std::uint32_t expected_format_version = 1;
    static constexpr std::uint32_t expected_feature_schema = 1;
    static constexpr std::uint32_t expected_num_features = 64 * 11 * 64;

    void load(const std::filesystem::path& path);

    bool loaded() const noexcept {
        return loaded_;
    }

    std::uint32_t accumulator_size() const noexcept {
        return accumulator_size_;
    }

    const float* feature_weights(std::uint32_t feature) const;

    std::span<const float> accumulator_bias() const {
        return accumulator_bias_;
    }

    float forward(
        std::span<const float> white_accumulator,
        std::span<const float> black_accumulator,
        Colour side_to_move
    ) const;

private:
    bool loaded_{false};
    
    std::uint32_t accumulator_size_{};
    std::uint32_t hidden1_size_{};
    std::uint32_t hidden2_size_{};
    std::uint32_t output_size_{};

    std::vector<float> feature_weights_;
    std::vector<float> accumulator_bias_;

    std::vector<float> layer1_weights_;
    std::vector<float> layer1_bias_;

    std::vector<float> layer2_weights_;
    std::vector<float> layer2_bias_;

    std::vector<float> output_weights_;
    std::vector<float> output_bias_;

};

} // namespace chess::engine
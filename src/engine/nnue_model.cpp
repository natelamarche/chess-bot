#include "engine/nnue_model.h"

#include <array>
#include <bit>
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace chess::engine {

namespace {

constexpr std::array<char, 8> expected_magic{
    'C', 'H', 'N', 'N', 'U', 'E', '\0', '\0'
};

void read_exact(
    std::ifstream& input,
    char* destination,
    std::size_t bytes,
    const char* description
) {
    input.read(
        destination,
        static_cast<std::streamsize>(bytes)
    );

    if (!input) {
        throw std::runtime_error(
            std::string("Failed to read ") + description
        );
    }

}

std::uint32_t read_u32_le(std::ifstream& input) {
    std::array<unsigned char, 4>bytes{};
    read_exact(
        input,
        reinterpret_cast<char*>(bytes.data()),
        bytes.size(),
        "uint32"
    );

    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::size_t checked_product(
    std::uint32_t first,
    std::uint32_t second
){
    const auto result = static_cast<std::uint64_t>(first) * second;

    if (result > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("NNUE tensor is too large");
    }

    return static_cast<std::size_t>(result);

}

void read_float_vector(
    std::ifstream& input,
    std::vector<float>& destination,
    std::size_t count,
    const char* description
){
    if constexpr (std::endian::native == std::endian::big) {
        throw std::runtime_error(
            "Big-endian systems are not currently supported"
        );
    }
    
    static_assert(sizeof(float) == 4);

    destination.resize(count);

    read_exact(
        input,
        reinterpret_cast<char*>(destination.data()),
        count * sizeof(float),
        description
    );
    
}

void linear(
    std::span<const float> input,
    std::span<const float> weights,
    std::span<const float> bias,
    std::span<float> output
)
{
    if (weights.size() != input.size() * output.size()){
        throw std::invalid_argument("Invalid linear weights dimensions");
    }

    if (bias.size() != output.size()){
        throw std::invalid_argument("Invalid linear bias dimensions");
    }

    for (std::size_t row = 0; row < output.size(); row++) {
        float sum = bias[row];

        for (std::size_t column = 0; column < input.size(); column++){
            sum += weights[row * input.size() + column]
            * input[column];
        }

        output[row] = sum;

    }

}

void relu_in_place(std::span<float> values) {
    for (float& value : values){
        value = std::max(value, 0.0F);
    }
}

} // namespace

void NnueModel::load(const std::filesystem::path& path){
    loaded_ = false;

    std::ifstream input(path, std::ios::binary);
    if (!input){
        throw std::runtime_error(
            "Could not open NNUE file: " + path.string()
        );
    }

    std::array<char, 8> magic{};
    read_exact(input, magic.data(), magic.size(), "NNUE magic");

    if (magic != expected_magic) {
        throw std::runtime_error("Invalid NNUE magic");
    }

    const std::uint32_t format_version = read_u32_le(input);
    const std::uint32_t feature_schema = read_u32_le(input);
    const std::uint32_t num_features = read_u32_le(input);

    accumulator_size_ = read_u32_le(input);
    hidden1_size_ = read_u32_le(input);
    hidden2_size_ = read_u32_le(input);
    output_size_ = read_u32_le(input);

    if (format_version != expected_format_version){
        throw std::runtime_error("Unsupported NNUE format version");
    }

    if (feature_schema != expected_feature_schema){
        throw std::runtime_error("Unsupported NNUE feature schema");
    }

    if (num_features != expected_num_features){
        throw std::runtime_error("Unexpected feature count");
    }

    if (
        accumulator_size_ != 128 ||
        hidden1_size_ != 64 ||
        hidden2_size_ != 32 ||
        output_size_ != 1
    ) {
        throw std::runtime_error("Unsopported NNUE architecture");
    }

    read_float_vector(
        input,
        feature_weights_,
        checked_product(num_features, accumulator_size_),
        "feature weights"
    );

    read_float_vector(
        input,
        accumulator_bias_,
        accumulator_size_,
        "accumulator bias"
    );

    read_float_vector(
        input,
        layer1_weights_,
        checked_product(accumulator_size_ * 2, hidden1_size_),
        "layer 1 weights"
    );

    read_float_vector(
        input,
        layer1_bias_,
        hidden1_size_,
        "layer 1 bias"
    );

    read_float_vector(
        input,
        layer2_weights_,
        checked_product(hidden1_size_, hidden2_size_),
        "layer 2 weights"
    );

    read_float_vector(
        input,
        layer2_bias_,
        hidden2_size_,
        "layer 2 bias"
    );

    read_float_vector(
        input,
        output_weights_,
        checked_product(hidden2_size_, output_size_),
        "output weights"
    );

    read_float_vector(
        input,
        output_bias_,
        output_size_,
        "output bias"
    );
    
    char trailing_byte{};
    if (input.read(&trailing_byte, 1)){
        throw std::runtime_error(
            "Unexpected data after NNUE tensors"
        );
    }

    if (!input.eof()){
        throw std::runtime_error(
            "Error while checking end of NNUE file"
        );
    }

    loaded_ = true;

}

const float* NnueModel::feature_weights(std::uint32_t feature) const {
    if (!loaded_) {
        throw std::logic_error("NNUE model is not loaded");
    }

    if (feature >= expected_num_features) {
        throw std::out_of_range("NNUE features index");
    }
    
    return feature_weights_.data()
        + static_cast<std::size_t>(feature) * accumulator_size_;

}

float NnueModel::forward(
        std::span<const float> white_accumulator,
        std::span<const float> black_accumulator,
        Colour side_to_move
    ) const {

    if (!loaded_) {
        throw std::logic_error("NNUE model is not loaded");
    }

    if (
        white_accumulator.size() != accumulator_size_ ||
        black_accumulator.size() != accumulator_size_
    ) {
        throw std::invalid_argument("Invalid accumulator size");
    }

    std::array<float, 256> input{};
    std::array<float, 64> hidden1{};
    std::array<float, 32> hidden2{};
    std::array<float, 1> output{};

    const auto first =
        side_to_move == Colour::White
            ? white_accumulator
            : black_accumulator;

    const auto second =
        side_to_move == Colour::White
            ? black_accumulator
            : white_accumulator;

    for (std::size_t i = 0; i < accumulator_size_; ++i) {
        input[i] = std::max(first[i], 0.0F);
        input[accumulator_size_ + i] =
            std::max(second[i], 0.0F);
    }

    linear(input, layer1_weights_, layer1_bias_, hidden1);
    relu_in_place(hidden1);

    linear(hidden1, layer2_weights_, layer2_bias_, hidden2);
    relu_in_place(hidden2);

    linear(hidden2, output_weights_, output_bias_, output);

    return output[0];
}


} // namespace chess::engine

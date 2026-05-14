#pragma once
#include <cstdint>

namespace chess {

    using Square = std::uint8_t;

    enum class Colour : std::uint8_t {White, Black};
    enum class Piece : std::int8_t {None, Pawn, Knight, Bishop, Rook, Queen, King};

    struct PieceOnSquare {
        Piece piece{Piece::None};
        Colour colour{Colour::White};        
    };

} // namespace chess
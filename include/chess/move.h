#pragma once
#include "types.h"

namespace chess {

struct Move {
    int from{};
    int to{};

    Piece promotion{Piece::None};

    bool is_en_passant{false};
    bool is_capture{false};
    bool is_castle{false};

};

} // namespace chess

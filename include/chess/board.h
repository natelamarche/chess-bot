#pragma once
#include <array>
#include <string>
#include "chess/types.h"

namespace chess {

class Board {
public:
    Board();

    void set_start_position();
    void set_fen(const std::string& fen);

    std::string to_string() const;

    const PieceOnSquare& at(Square sq) const {return squares_[sq];}

    Colour side_to_move() const {return side_to_move_;}
    int en_passant_suquare() const {return en_passant_sq_;} // -1 if none

    bool can_castle_white_k() const {return castle_wk_;}
    bool can_castle_white_q() const {return castle_wq_;}
    bool can_castle_black_k() const {return castle_bk_;}
    bool can_castle_black_q() const {return castle_bq_;}


private:
    std::array<PieceOnSquare, 64> squares_{};
    Colour side_to_move_{Colour::White};
    bool castle_wk_{false}, castle_wq_{false}, castle_bk_{false}, castle_bq_{false};
    int en_passant_sq_{-1};


};

} // namespace chess

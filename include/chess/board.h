#pragma once
#include <array>
#include <string>
#include "chess/types.h"
#include "chess/move.h"

namespace chess {

struct CastlingRights {
    bool castle_wk{false}, castle_wq{false}, castle_bk{false}, castle_bq{false};
};

struct UndoState {
    Piece captured_piece{Piece::None};
    CastlingRights castling_rights{};
    int en_passant_sq{-1};

};

class Board {
public:
    Board();

    void set_start_position();
    void set_fen(const std::string& fen);

    std::string to_string() const;

    const PieceOnSquare& at(Square sq) const {return squares_[sq];}
    const UndoState make_move(const Move& move);
    void unmake_move(const Move& move, const UndoState& undo_state);

    Colour side_to_move() const {return side_to_move_;}
    int en_passant_square() const {return en_passant_sq_;} // -1 if none

    bool can_castle_white_k() const {return castling_rights_.castle_wk;}
    bool can_castle_white_q() const {return castling_rights_.castle_wq;}
    bool can_castle_black_k() const {return castling_rights_.castle_bk;}
    bool can_castle_black_q() const {return castling_rights_.castle_bq;}


private:
    std::array<PieceOnSquare, 64> squares_{};
    Colour side_to_move_{Colour::White};
    CastlingRights castling_rights_{};
    int en_passant_sq_{-1};
    
    void clear_castling_rights_for_square(Square sq);

};

} // namespace chess

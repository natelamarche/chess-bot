#include "chess/board.h"
#include <sstream>

namespace chess {

// -------------Helper functions------------- //
static bool is_file(char c) { return c >= 'a' && c <= 'h'; }
static bool is_rank(char c) { return c >= '1' && c <= '8'; }

static int square_from_algebraic(const std::string& s) {
    // "e4" -> file=4, rank=3 -> sq = 3*8 + 4
    if (s.size() != 2 || !is_file(s[0]) || !is_rank(s[1])) return -1;
    int file = s[0] - 'a';
    int rank = s[1] - '1';
    return rank * 8 + file;
}

static Piece piece_from_fen_char(char c) {
    switch (std::tolower(static_cast<unsigned char>(c))) {
        case 'p': return Piece::Pawn;
        case 'n': return Piece::Knight;
        case 'b': return Piece::Bishop;
        case 'r': return Piece::Rook;
        case 'q': return Piece::Queen;
        case 'k': return Piece::King;
        default:  return Piece::None;
    }
}


static char piece_char(Piece p, Colour c){
    char ch = '.';

    switch (p){
        case Piece::Pawn:   ch = 'p'; break;
        case Piece::Knight: ch = 'n'; break;
        case Piece::Bishop: ch = 'b'; break;
        case Piece::Rook:   ch = 'r'; break;
        case Piece::Queen:  ch = 'q'; break;
        case Piece::King:   ch = 'k'; break;
        default: return '.';
    }

    if (c == Colour::White) ch = static_cast<char>(ch - 'a' + 'A'); // uppercase for White

    return ch;

}

// ------------------------------------------- //

Board::Board(){
    set_start_position();
}

void Board::set_start_position(){
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

std::string Board::to_string() const {
    std::ostringstream out;
    out << "  +-----------------+\n";
    for (int rank = 7; rank >= 0; rank--){
        out << rank + 1 << " | ";
        for (int file = 0; file < 8; file++){
            int sq = rank * 8 + file;
            const auto& ps = squares_[sq];
            out << piece_char(ps.piece, ps.colour) << ' ';
        }
        out << "\n";
    }
    out << "  +-----------------+\n";
    out << "    a b c d e f g h\n";
    return out.str();
}

void Board::set_fen(const std::string& fen) {
    // Expected: "piecePlacement side castling ep halfmove fullmove"
    // Example:  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    std::istringstream iss(fen);
    std::string placement, stm, castling, ep;
    int halfmove = 0, fullmove = 1;

    if (!(iss >> placement >> stm >> castling >> ep >> halfmove >> fullmove)) {
        throw std::runtime_error("Invalid FEN: expected 6 fields");
    }

    // Clear board
    squares_.fill({Piece::None, Colour::White});

    // ---- 1) Piece placement ----
    // placement has 8 ranks separated by '/', starting from rank 8 down to rank 1
    int rank = 7; // start at rank 8
    int file = 0;

    for (char c : placement) {
        if (c == '/') {
            if (file != 8) throw std::runtime_error("Invalid FEN: rank does not have 8 files");
            rank--;
            file = 0;
            continue;
        }

        if (std::isdigit(static_cast<unsigned char>(c))) {
            int empty = c - '0';
            if (empty < 1 || empty > 8) throw std::runtime_error("Invalid FEN: bad digit in placement");
            file += empty;
            if (file > 8) throw std::runtime_error("Invalid FEN: too many files in rank");
            continue;
        }

        Piece p = piece_from_fen_char(c);
        if (p == Piece::None) throw std::runtime_error("Invalid FEN: unknown piece character");

        if (file >= 8 || rank < 0) throw std::runtime_error("Invalid FEN: placement overflow");

        Colour col = std::isupper(static_cast<unsigned char>(c)) ? Colour::White : Colour::Black;
        int sq = rank * 8 + file; // rank 7..0 maps to 8th..1st
        squares_[sq] = {p, col};
        file++;
    }

    if (rank != 0 || file != 8) {
        // After parsing, we should have ended exactly at rank 1 with 8 files
        throw std::runtime_error("Invalid FEN: wrong number of ranks/files");
    }

    // ---- 2) Side to move ----
    if (stm == "w") side_to_move_ = Colour::White;
    else if (stm == "b") side_to_move_ = Colour::Black;
    else throw std::runtime_error("Invalid FEN: side-to-move must be 'w' or 'b'");

    // ---- 3) Castling rights ----
    castling_rights_.castle_wk = castling_rights_.castle_wq = castling_rights_.castle_bk = castling_rights_.castle_bq = false;
    if (castling != "-") {
        for (char c : castling) {
            switch (c) {
                case 'K': castling_rights_.castle_wk = true; break;
                case 'Q': castling_rights_.castle_wq = true; break;
                case 'k': castling_rights_.castle_bk = true; break;
                case 'q': castling_rights_.castle_bq = true; break;
                default: throw std::runtime_error("Invalid FEN: bad castling rights");
            }
        }
    }

    // ---- 4) En passant ----
    if (ep == "-") {
        en_passant_sq_ = -1;
    } else {
        int sq = square_from_algebraic(ep);
        if (sq < 0) throw std::runtime_error("Invalid FEN: bad en passant square");
        en_passant_sq_ = sq;
    }

    // halfmove/fullmove currently ignored by engine logic, but we parsed them for validity.
}

void Board::clear_castling_rights_for_square(Square sq){
    switch (sq) {
        case 0:
            castling_rights_.castle_wq = false;
            break;
        case 4:
            castling_rights_.castle_wk = false;
            castling_rights_.castle_wq = false;
            break;
        case 7:
            castling_rights_.castle_wk = false;
            break;
        case 56:
            castling_rights_.castle_bq = false;
            break;
        case 60:
            castling_rights_.castle_bq = false;
            castling_rights_.castle_bk = false;
            break;
        case 63:
            castling_rights_.castle_bk = false;
            break;
    }
}


UndoState Board::make_move(const Move& move){
    
    Colour stm = side_to_move_;
    PieceOnSquare from_piece = squares_[move.from]; 
    UndoState undo_state{squares_[move.to].piece, castling_rights_, en_passant_sq_};

    // Change castling rights
    if (castling_rights_.castle_bk || castling_rights_.castle_bq || castling_rights_.castle_wk || castling_rights_.castle_wq){
        clear_castling_rights_for_square(move.from);
        clear_castling_rights_for_square(move.to);
    }

    // Handle en passant
    en_passant_sq_ = -1;
    if (from_piece.piece == Piece::Pawn && std::abs(move.from - move.to) == 16){
        en_passant_sq_ = (move.from + move.to) / 2;
    }

    // Make move
    squares_[move.from] = {};

    if (move.promotion != Piece::None){
        squares_[move.to] = {move.promotion, stm};

    } else if (move.is_en_passant){
        undo_state.captured_piece = Piece::Pawn;
        int captured_pawn_sq = move.to + (stm == Colour::White ? -8 : 8);
        squares_[move.to] = from_piece;
        squares_[captured_pawn_sq] = {};
        

    } else if (move.is_castle){
        if (move.from > move.to){
            squares_[move.to] = from_piece;
            squares_[move.to + 1] = {Piece::Rook, stm};
            squares_[move.to - 2] = {};

        } else {
            squares_[move.to] = from_piece;
            squares_[move.to - 1] = {Piece::Rook, stm};
            squares_[move.to + 1] = {};

        }
    } else {
        squares_[move.to] = from_piece;

    }


    side_to_move_ = (side_to_move_ == Colour::White ? Colour::Black : Colour::White);

    return undo_state;

}

void Board::unmake_move(const Move& move, const UndoState& undo_state){
    castling_rights_ = undo_state.castling_rights;
    en_passant_sq_ = undo_state.en_passant_sq;

    Colour o_stm = side_to_move_;
    side_to_move_ = (side_to_move_ == Colour::White ? Colour::Black : Colour::White);
    Colour stm = side_to_move_;

    if (move.is_en_passant){
        squares_[move.from] = {Piece::Pawn, stm};
        squares_[move.to] = {};
        squares_[move.to + (stm == Colour::White ? -8 : 8)] = {Piece::Pawn, o_stm};
    
    } else if (move.promotion != Piece::None) {
        squares_[move.from] = {Piece::Pawn, stm};
        squares_[move.to] = {undo_state.captured_piece, o_stm};

    } else if (move.is_castle){
        if (move.from > move.to){
            squares_[move.from] = {Piece::King, stm};
            squares_[move.to] = {Piece::None, stm};
            squares_[move.to + 1] = {};
            squares_[move.to - 2] = {Piece::Rook, stm};

        } else {
            squares_[move.from] = {Piece::King, stm};
            squares_[move.to] = {Piece::None, stm};
            squares_[move.to - 1] = {};
            squares_[move.to + 1] = {Piece::Rook, stm};

        }
    } else {
        squares_[move.from] = squares_[move.to];
        squares_[move.to] = {undo_state.captured_piece, o_stm};
    }
    
};


} // namespace chess

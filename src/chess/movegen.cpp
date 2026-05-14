#include "chess/movegen.h"
#include <array>

namespace chess {

namespace {

struct PawnMoveParams {
    int forward;
    int starting_rank;
    int promotion_rank;
    int ep_sq;
};

void add_sliding_moves(const Board& board,
                       Square sq,
                       Colour stm,
                       const std::array<std::array<int, 2>, 4>& directions,
                       std::vector<Move>& moves){
    int rank = sq / 8;
    int file = sq % 8;

    for (const auto& dir : directions){
        int r = rank;
        int f = file;

        while (true){
            r += dir[1];
            f += dir[0];

            if (r < 0 || r > 7 || f < 0 || f > 7) break;

            int new_sq = r * 8 + f;
            const auto& target = board.at(static_cast<Square>(new_sq));

            if (target.piece == Piece::None){
                moves.push_back({sq, new_sq});
            } else {
                if (target.colour != stm){
                    Move m{sq, new_sq};
                    m.is_capture = true;
                    moves.push_back(m);
                }
                break;
            }
        }
    }
}


void generate_pawn_moves(const Board& board, std::vector<Move>& moves, Square sq, Colour stm, const PawnMoveParams& pmp){

    static const std::array<Piece, 4> promotion_pieces
        {{Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight}};

    auto add_promotion_moves = [&](Square from, Square to, bool is_capture){
        for (Piece p : promotion_pieces){
            Move m{from, to};
            m.promotion = p;
            m.is_capture = is_capture;
            moves.push_back(m);
        }
    };

    int rank = sq/8;
    int file = sq%8;

    int to = sq + pmp.forward;

    if (to < 64 && to >= 0 && board.at(to).piece == Piece::None){
        if (to / 8 == pmp.promotion_rank){
            add_promotion_moves(sq, to, false);
        } else {
            moves.push_back({sq, static_cast<Square>(to)});

            if (rank == pmp.starting_rank){
                int to2 = to + pmp.forward;

                if (board.at(to2).piece == Piece::None){
                    moves.push_back({sq, static_cast<Square>(to2)});
                }
            }
        }
    }    

    for (int df: {-1, 1}){
        int f = file + df;
        if (f < 0 || f > 7) continue;

        int cap_sq = sq + df + pmp.forward;
        if (cap_sq < 0 || cap_sq >= 64) continue;

        const auto& target = board.at(cap_sq);

        if (target.piece != Piece::None && target.colour != stm){
            if (cap_sq / 8 == pmp.promotion_rank){
                add_promotion_moves(sq, cap_sq, true);
            } else {
                Move m{sq, static_cast<Square>(cap_sq)};
                m.is_capture = true;
                moves.push_back(m);
            }
        }

        if (cap_sq == pmp.ep_sq){
            Move m{sq, static_cast<Square>(cap_sq)};
            m.is_capture = true;
            m.is_en_passant = true;
            moves.push_back(m);
        }
    }


}

void generate_knight_moves(const Board& board, std::vector<Move>& moves, Square sq, Colour stm){

    static const std::array<std::array<int, 2>, 8> knight_moveset 
            {{{-2, 1}, {-2, -1}, {-1, -2}, {-1, 2}, 
            {1, -2}, {1, 2}, {2, 1}, {2, -1}}};

    int file = sq%8;

    for (const auto& dsq: knight_moveset){
        if (file + dsq[0] < 0 || file + dsq[0] > 7) continue;

        int to = sq + dsq[0] + 8 * dsq[1];

        if (to < 0 || to >= 64) continue;

        const auto& target = board.at(to);

        if (target.piece == Piece::None){
            moves.push_back({sq, to});

        } else if (target.colour != stm){
            Move m{sq, to};
            m.is_capture = true;
            moves.push_back(m);

        }
        
    }

}

void generate_rook_moves(const Board& board, std::vector<Move>& moves, Square sq, Colour stm){

    static const std::array<std::array<int, 2>, 4> rook_dirs
        {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

    add_sliding_moves(board, sq, stm, rook_dirs, moves);
    
}

void generate_bishop_moves(const Board& board, std::vector<Move>& moves, Square sq, Colour stm){

    static const std::array<std::array<int, 2>, 4> bishop_dirs
        {{{1, 1}, {-1, 1}, {1, -1}, {-1, -1}}};

    add_sliding_moves(board, sq, stm, bishop_dirs, moves);
    
}

void generate_queen_moves(const Board& board, std::vector<Move>& moves, Square sq, Colour stm){

    static const std::array<std::array<int, 2>, 4> rook_dirs
        {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
    static const std::array<std::array<int, 2>, 4> bishop_dirs
        {{{1, 1}, {-1, 1}, {1, -1}, {-1, -1}}};

    add_sliding_moves(board, sq, stm, rook_dirs, moves);
    add_sliding_moves(board, sq, stm, bishop_dirs, moves);
    
}

void generate_king_moves(const Board& board, std::vector<Move>& moves, Square sq, Colour stm){

    static const std::array<std::array<int, 2>, 8> king_moveset 
        {{{-1, -1}, {-1, 0}, {-1, 1}, {0, 1},
         {1, 1}, {1, 0}, {1, -1}, {0, -1}}};


    int rank = sq/8;
    int file = sq%8;

    for (const auto& dsq: king_moveset){
        int f = file + dsq[0];
        int r = rank + dsq[1];

        if (f < 0 || f > 7 || r < 0 || r > 7) continue;

        int to = r * 8 + f;
        const auto& target = board.at(to);

        if (target.piece == Piece::None){
            moves.push_back({sq, to});

        } else if (target.colour != stm){
            Move m {sq, to};
            m.is_capture = true;
            moves.push_back(m);

        }

    }

    if (stm == Colour::White){
        if (sq == 4 && board.can_castle_white_k()){
            if (board.at(5).piece == Piece::None && board.at(6).piece == Piece::None){
                const auto& rook_sq = board.at(7);
                if (rook_sq.piece == Piece::Rook && rook_sq.colour == stm){
                    Move m{sq, 6};
                    m.is_castle = true;
                    moves.push_back(m);
                }
            }
        }
        if (sq == 4 && board.can_castle_white_q()){
            if (board.at(1).piece == Piece::None &&
                board.at(2).piece == Piece::None &&
                board.at(3).piece == Piece::None){
                const auto& rook_sq = board.at(0);
                if (rook_sq.piece == Piece::Rook && rook_sq.colour == stm){
                    Move m{sq, 2};
                    m.is_castle = true;
                    moves.push_back(m);
                }
            }
        }

    } else {
        if (sq == 60 && board.can_castle_black_k()){
            if (board.at(61).piece == Piece::None && board.at(62).piece == Piece::None){
                const auto& rook_sq = board.at(63);
                if (rook_sq.piece == Piece::Rook && rook_sq.colour == stm){
                    Move m{sq, 62};
                    m.is_castle = true;
                    moves.push_back(m);
                }
            }
        }
        if (sq == 60 && board.can_castle_black_q()){
            if (board.at(57).piece == Piece::None &&
                board.at(58).piece == Piece::None &&
                board.at(59).piece == Piece::None){
                const auto& rook_sq = board.at(56);
                if (rook_sq.piece == Piece::Rook && rook_sq.colour == stm){
                    Move m{sq, 58};
                    m.is_castle = true;
                    moves.push_back(m);
                }
            }
        }
    }
}

bool is_square_attacked(const Board& board, Square sq, Colour by_colour){
    
    static const std::array<std::array<int, 2>, 8> knight_moveset 
            {{{-2, 1}, {-2, -1}, {-1, -2}, {-1, 2}, 
            {1, -2}, {1, 2}, {2, 1}, {2, -1}}};

    static const std::array<std::array<int, 2>, 4> rook_dirs
        {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

    static const std::array<std::array<int, 2>, 4> bishop_dirs
        {{{1, 1}, {-1, 1}, {1, -1}, {-1, -1}}};
    
    static const std::array<std::array<int, 2>, 8> king_moveset 
        {{{-1, -1}, {-1, 0}, {-1, 1}, {0, 1},
         {1, 1}, {1, 0}, {1, -1}, {0, -1}}};

    static const std::array<std::array<int, 2>, 2> wPawn_moveset 
        {{{-1, -1}, {1, -1}}};

    static const std::array<std::array<int, 2>, 2> bPawn_moveset 
        {{{-1, 1}, {1, 1}}};

    const int file = sq % 8;
    const int rank = sq / 8;

    // Attack by pawn
    if (by_colour == Colour::White){
        for (const auto& dsq : wPawn_moveset){
            int f = file + dsq[0];
            if (f > 7 || f < 0) continue;
            
            int attacker_sq = f + (rank + dsq[1]) * 8;
            if (attacker_sq > 63 || attacker_sq < 0) continue;

            const auto& attacker = board.at(attacker_sq);
            if (attacker.piece == Piece::Pawn && attacker.colour == by_colour) return true;

        }
    } else {
        for (const auto& dsq : bPawn_moveset){
            int f = file + dsq[0];
            if (f > 7 || f < 0) continue;
            
            int attacker_sq = f + (rank + dsq[1]) * 8;
            if (attacker_sq > 63 || attacker_sq < 0) continue;

            const auto& attacker = board.at(attacker_sq);
            if (attacker.piece == Piece::Pawn && attacker.colour == by_colour) return true;

        }
    }
    
    // Attack by knight
    for (const auto& dsq : knight_moveset){
        int f = file + dsq[0];
        if (f > 7 || f < 0) continue;
        
        int attacker_sq = f + (rank + dsq[1]) * 8;
        if (attacker_sq > 63 || attacker_sq < 0) continue;

        const auto& attacker = board.at(attacker_sq);
        if (attacker.piece == Piece::Knight && attacker.colour == by_colour) return true;


    }

    // Attack by king
    for (const auto& dsq : king_moveset){
        int f = file + dsq[0];
        if (f > 7 || f < 0) continue;
        
        int attacker_sq = f + (rank + dsq[1]) * 8;
        if (attacker_sq > 63 || attacker_sq < 0) continue;

        const auto& attacker = board.at(attacker_sq);
        if (attacker.piece == Piece::King && attacker.colour == by_colour) return true;


    }

    // Attack by rook/queen
    for (const auto& dsq : rook_dirs){
        int r = rank;
        int f = file;

        while (true){
            r += dsq[1];
            f += dsq[0];

            if (r < 0 || r > 7 || f < 0 || f > 7) break;

            int new_sq = r * 8 + f;
            const auto& target = board.at(static_cast<Square>(new_sq));

            if (target.piece == Piece::None){
                continue;
            } else {
                if (target.colour == by_colour && (target.piece == Piece::Rook || target.piece == Piece::Queen)){
                    return true;
                }
                break;
            }
        }
    }

    // Attack by bishop/queen
    for (const auto& dsq : bishop_dirs){
        int r = rank;
        int f = file;

        while (true){
            r += dsq[1];
            f += dsq[0];

            if (r < 0 || r > 7 || f < 0 || f > 7) break;

            int new_sq = r * 8 + f;
            const auto& target = board.at(static_cast<Square>(new_sq));

            if (target.piece == Piece::None){
                continue;
            } else {
                if (target.colour == by_colour && (target.piece == Piece::Bishop || target.piece == Piece::Queen)){
                    return true;
                }
                break;
            }
        }
    }

    return false;

}

bool in_check(const Board& board, Colour clr){
    
    for (Square sq = 0; sq < 64; sq++){
        const auto& target = board.at(sq);
        
        if (target.colour == clr && target.piece == Piece::King) {
            Colour attacker = (clr == Colour::White ? Colour::Black : Colour::White);

            return is_square_attacked(board, sq, attacker);
        }
    } 

    return false;
    
}

}

std::vector<Move> generate_pseudo_legal_moves(const Board& board){
    std::vector<Move> moves;

    const Colour stm = board.side_to_move();

    const PawnMoveParams pmp{
        (stm == Colour::White) ? 8 : -8,
        (stm == Colour::White) ? 1 : 6,
        (stm == Colour::White) ? 7 : 0,
        board.en_passant_square()
    };

    for (Square sq = 0; sq < 64; sq++){
        const auto& ps = board.at(sq);
        if (ps.piece != Piece::None && ps.colour == stm) {
            switch (ps.piece) {
                case Piece::Pawn:
                    generate_pawn_moves(board, moves, sq, stm, pmp);
                    break;
                case Piece::Knight:
                    generate_knight_moves(board, moves, sq, stm);
                    break;
                case Piece::Rook:
                    generate_rook_moves(board, moves, sq, stm);
                    break;
                case Piece::Bishop:
                    generate_bishop_moves(board, moves, sq, stm);
                    break;
                case Piece::Queen:
                    generate_queen_moves(board, moves, sq, stm);
                    break;
                case Piece::King:
                    generate_king_moves(board, moves, sq, stm);
                    break;
                default:
                    break;
                
            }
        }
    }

    return moves;

}

std::vector<Move> generate_legal_moves(const Board& board){
    std::vector<Move> legal;
    std::vector<Move> pseudo_legal = generate_pseudo_legal_moves(board);

    Colour stm = board.side_to_move();

    for (const Move& move : pseudo_legal){
        if (move.is_castle) {
            Colour enemy = (stm == Colour::White) ? Colour::Black : Colour::White;

            if (in_check(board, stm)) continue;

            Square transit = (move.to > move.from) 
                ? static_cast<Square>(move.from + 1)
                : static_cast<Square>(move.from - 1);

            if (is_square_attacked(board, transit, enemy)) continue;

        }
        
        Board next = board;
        next.make_move(move);

        if (!in_check(next, stm)) {
            legal.push_back(move);
        }
    }

    return legal;

}

}

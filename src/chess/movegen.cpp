#include "chess/movegen.h"

namespace chess {
    std::vector<Move> generate_pawn_moves(const Board& board){
        std::vector<Move> moves;

        Colour stm = board.side_to_move();
        int forward = (stm == Colour::White) ? 8 : -8;
        int start_rank = (stm == Colour::White) ? 1 : 6;
        int ep_sq = board.en_passant_square();

        for (Square sq = 0; sq < 64; sq++){
            const auto& ps = board.at(sq);

            if (ps.piece != Piece::Pawn || ps.colour != stm){
                continue;
            }

            int rank = sq/8;
            int file = sq%8;

            int to = sq + forward;

            if (to < 64 && to >= 0 && board.at(to).piece == Piece::None){
                moves.push_back({sq, static_cast<Square>(to)});

                if (rank == start_rank){
                    int to2 = to + forward;

                    if (board.at(to2).piece == Piece::None){
                        moves.push_back({sq, static_cast<Square>(to2)});
                    }
                }
            }

            for (int df: {-1, 1}){
                int f = file + df;
                if (f < 0 || f > 7) continue;

                int cap_sq = sq + df + forward;
                if (cap_sq < 0 || cap_sq >= 64) continue;

                const auto& target = board.at(cap_sq);
                if (target.piece != Piece::None && target.colour != stm){
                    Move m{sq, static_cast<Square>(cap_sq)};
                    m.is_capture = true;
                    moves.push_back(m);
                }

                if (cap_sq == ep_sq){
                    Move m{sq, static_cast<Square>(cap_sq)};
                    m.is_capture = true;
                    m.is_en_passant = true;
                    moves.push_back(m);
                }
            }

        }

        return moves;
    }



}

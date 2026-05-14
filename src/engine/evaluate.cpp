#include "engine/evaluate.h"
#include <array>

namespace chess::engine {
namespace {

int piece_value(Piece piece){
    switch (piece) {
        case Piece::Pawn: return 100;
        case Piece::Knight: return 300;
        case Piece::Bishop: return 300;
        case Piece::Rook: return 500;
        case Piece::Queen: return 900;
        case Piece::King: return 0;
        case Piece::None: return 0;
    }

    return 0;

}

int material_eval(const Board& board){
    int score = 0;

    for (Square sq = 0; sq < 64; sq++){
        const PieceOnSquare& target = board.at(sq);
        
        if (target.colour == Colour::White){
            score += piece_value(target.piece);
        } else {
            score -= piece_value(target.piece);
        }
    }

    return score;

}


}

int evaluate(const Board& board){

    int score = 0;

    score += material_eval(board);

    return board.side_to_move() == Colour::White ? score : -score;

}

}
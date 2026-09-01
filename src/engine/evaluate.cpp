#include "engine/evaluate.h"
#include "engine/nnue_state.h"
#include <array>
#include <cmath>

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

int evaluate(const Board& board, const NnueModel& model){

    const auto white = nnue::rebuild_accumulator(board, Colour::White, model);
    const auto black = nnue::rebuild_accumulator(board, Colour::Black, model);

    const float pawns = model.forward(white, black, board.side_to_move());

    return static_cast<int>(std::lround(pawns * 100.0F));

}

}
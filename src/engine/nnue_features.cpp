#include "engine/nnue_features.h"
#include <optional>
#include <stdexcept>

namespace chess::engine::nnue {

namespace {

constexpr std::uint32_t num_piece_types = 11;

std::uint32_t relative_piece_index(PieceOnSquare piece, Colour perspective){
    bool is_friendly = piece.colour == perspective;
    
    if (is_friendly){
        switch (piece.piece) {
            case Piece::Pawn: return 0;
            case Piece::Knight: return 1;
            case Piece::Bishop: return 2;
            case Piece::Rook: return 3;
            case Piece::Queen: return 4;
            case Piece::King:
                throw std::logic_error("Friendly king is not encoded");
            case Piece::None:
                throw std::logic_error("Empty square cannot be encoded");
        }
    } else {
        switch (piece.piece) {
            case Piece::Pawn: return 5;
            case Piece::Knight: return 6;
            case Piece::Bishop: return 7;
            case Piece::Rook: return 8;
            case Piece::Queen: return 9;
            case Piece::King: return 10;
            case Piece::None:
                throw std::logic_error("Empty square cannot be encoded");
        }
    }

    throw std::logic_error("Invalid piece type");
}

Square orient_square(Square square, Colour perspective){
    if (perspective == Colour::White) {
        return square;
    } else {
        return square ^ 56;
    }
}

} // namespace

std::uint32_t feature_index(Square king_square, std::uint32_t piece_index, Square piece_square){
    return king_square * num_piece_types * 64
            + piece_index * 64
            + piece_square;
}

std::vector<std::uint32_t> active_features(const Board& board, Colour perspective){
    std::optional<Square> king_square;
    for (Square sq = 0; sq <= 63; sq++){
        const PieceOnSquare& pos = board.at(sq);
        if (pos.piece == Piece::King && pos.colour == perspective ) {
            king_square = orient_square(sq, perspective);
            break;
        }
    }
    
    if (!king_square) {
        throw std::invalid_argument(
            "Cannot encode position without persepective king"
        );
    }


    std::vector<std::uint32_t> features{};
    features.reserve(31);

    for (Square sq = 0; sq <= 63; sq++){
        const PieceOnSquare& pos = board.at(sq);
        if (pos.piece == Piece::None
        || (pos.piece == Piece::King && pos.colour == perspective)) continue;
        
        std::uint32_t piece_index = relative_piece_index(pos, perspective);

        Square piece_square = orient_square(sq, perspective);
        
        features.push_back(feature_index(*king_square, piece_index, piece_square));
    }
    
    return features;

}


} // namespace chess::engine::nnue
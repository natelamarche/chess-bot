#include "engine/nnue_state.h"
#include <algorithm>
#include <stdexcept>
#include "engine/nnue_features.h"

namespace chess::engine {

namespace {

Accumulator build_accumulator(const Board& board, Colour perspective, const NnueModel& model) {
    if (!model.loaded()) {
        throw std::logic_error("NNUE model is not loaded");
    }

    const auto bias = model.accumulator_bias();

    Accumulator accumulator{};

    std::copy(bias.begin(), bias.end(), accumulator.begin());

    for (std::uint32_t feature : nnue::active_features(board, perspective)){
        const float* weights = model.feature_weights(feature);

        for (std::size_t i = 0; i < accumulator.size(); i++) {
            accumulator[i] += weights[i];
        }

    }

    return accumulator;

}

void update_perspective(
    Accumulator& accumulator,
    Colour perspective,
    Square king_square,
    const MoveFeatureChanges& changes,
    const NnueModel& model
) {
    for (std::size_t n = 0; n < changes.removed_count; n++) {
        const PieceChange& change = changes.removed[n];

        if (
            change.piece.piece == Piece::King &&
            change.piece.colour == perspective
        ) {
            continue;
        }

        const auto feature = nnue::feature_for_piece(
            king_square, 
            change.piece, 
            change.square, 
            perspective
        );

        const float* weights = model.feature_weights(feature);

        for (std::size_t i = 0; i < accumulator.size(); i++){
            accumulator[i] -= weights[i];
        }
    }

    for (std::size_t n = 0; n < changes.added_count; n++) {
        const PieceChange& change = changes.added[n];

        if (
            change.piece.piece == Piece::King &&
            change.piece.colour == perspective
        ) {
            continue;
        }

        const auto feature = nnue::feature_for_piece(
            king_square, 
            change.piece, 
            change.square, 
            perspective
        );

        const float* weights = model.feature_weights(feature);

        for (std::size_t i = 0; i < accumulator.size(); i++){
            accumulator[i] += weights[i];
        }
    }

}

Square find_king(
    const Board& board,
    Colour colour
) {
    for (Square square = 0; square < 64; square++) {
        const PieceOnSquare& piece = board.at(square);

        if (
            piece.piece == Piece::King &&
            piece.colour == colour
        ) {
            return square;
        }

    }

    throw std::logic_error("Position has no king");
}

} // namespace


MoveFeatureChanges prepare_move(const Board& board, const Move& move){
    MoveFeatureChanges changes;

    const PieceOnSquare moving = board.at(move.from);

    if (moving.piece == Piece::None) {
        throw std::logic_error(
            "Cannot prepare a move from an empty square"
        );
    }

    changes.remove(moving, move.from);

    if (moving.piece == Piece::King) {
        changes.king_moved = true;
        changes.king_colour = moving.colour;
    }

    if (move.is_castle) {
        const bool queen_side = move.from > move.to;

        const Square rook_from = queen_side
            ? static_cast<Square>(move.to - 2)
            : static_cast<Square>(move.to + 1);

        const Square rook_to = queen_side 
            ? static_cast<Square>(move.to + 1)
            : static_cast<Square>(move.to - 1);

        const PieceOnSquare rook{
            Piece::Rook,
            moving.colour
        };

        changes.add(moving, move.to);
        changes.remove(rook, rook_from);
        changes.add(rook, rook_to);

        return changes;
    }

    if (move.is_en_passant) {
        const Square captured_square = 
            moving.colour == Colour::White
                ? static_cast<Square>(move.to - 8)
                : static_cast<Square>(move.to + 8);
        
        changes.remove(
            board.at(captured_square),
            captured_square
        );
    } else {
        const PieceOnSquare captured = board.at(move.to);

        if (captured.piece != Piece::None) {
            changes.remove(captured, move.to);
        }
    }

    const PieceOnSquare placed{
        move.promotion == Piece::None
            ? moving.piece
            : move.promotion,
        moving.colour        
    };

    changes.add(placed, move.to);

    return changes;
}

void NnueState::initialize(const Board& board, const NnueModel& model){
    if (initialized_){
        throw std::logic_error("State is already initialized");
    }

    white_ = build_accumulator(board, Colour::White, model);
    black_ = build_accumulator(board, Colour::Black, model);

    initialized_ = true;

}

void NnueState::apply_move(const MoveFeatureChanges& changes, const Board& board_after, const NnueModel& model){
    if (!initialized_){
        throw std::logic_error("NNUE state is not initialized");
    }
    
    if (changes.king_moved &&
        changes.king_colour == Colour::White){

        white_ = build_accumulator(
            board_after, 
            Colour::White, 
            model
        );

    } else {
        update_perspective(
            white_, 
            Colour::White, 
            find_king(board_after, Colour::White), 
            changes, 
            model
        );

    }

    if (changes.king_moved &&
        changes.king_colour == Colour::Black){

        black_ = build_accumulator(
            board_after, 
            Colour::Black, 
            model
        );
    
    } else {
        update_perspective(
            black_, 
            Colour::Black, 
            find_king(board_after, Colour::Black), 
            changes, 
            model
        );
    }
}


} // namespace chess::engine

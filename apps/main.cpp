#include <iostream>
#include <string>
#include <vector>
#include "chess/board.h"
#include "chess/movegen.h"
#include "engine/search.h"
#include "engine/nnue_model.h"

using namespace std;

std::string square_to_string(chess::Square sq) {
    int file = sq % 8;
    int rank = sq / 8;

    char fileChar = 'a' + file;
    char rankChar = '1' + rank;

    return std::string{fileChar, rankChar};
}

std::string colour_to_string(chess::Colour colour) {
    return colour == chess::Colour::White ? "White" : "Black";
}

chess::Colour opposite_colour(chess::Colour colour) {
    return colour == chess::Colour::White ? chess::Colour::Black : chess::Colour::White;
}

bool print_game_over_if_needed(const chess::Board& board, const vector<chess::Move>& moves) {
    if (!moves.empty()) {
        return false;
    }

    chess::Colour side_to_move = board.side_to_move();
    if (chess::in_check(board, side_to_move)) {
        cout << "Checkmate. " << colour_to_string(opposite_colour(side_to_move)) << " wins.\n";
    } else {
        cout << "Stalemate.\n";
    }

    return true;
}

int main() {
    chess::Board b;
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    cout << "To make a move, use the format: {xx xx}, no parentheses (e.g. e2 e4)\n";
    chess::engine::NnueModel model;
    model.load("ml/model/model.nnue");
    
    chess::engine::Searcher searcher{model};

    while (true) {

        cout << b.to_string() << "\n";

        vector<chess::Move> moves = chess::generate_legal_moves(b);
        if (print_game_over_if_needed(b, moves)) {
            break;
        }
        
        while (true){
            string from;
            string to;

            cin >> from >> to;

            chess::Square sq_f = (from[0] - 'a') + 8 * (from[1] - '1');
            chess::Square sq_t = (to[0] - 'a') + 8 * (to[1] - '1');

            auto it = moves.begin();
            while (it != moves.end() && (it->from != sq_f || it->to != sq_t)) it++;

            if (it == moves.end()) {
                continue;
            } else {
                b.make_move(*it);
                break;
            }
        }

        vector<chess::Move> opponent_moves = chess::generate_legal_moves(b);
        if (print_game_over_if_needed(b, opponent_moves)) {
            cout << b.to_string() << "\n";
            break;
        }

        chess::engine::SearchResult opponent_move = searcher.search_best_move(b, 5);

        b.make_move(opponent_move.move);

        string from = square_to_string(opponent_move.move.from);
        string to = square_to_string(opponent_move.move.to);

        cout << "\n\n" << from << " --> " << to << " Evaluation: " << -opponent_move.score << "\n";
   
    }

    return 0;
}

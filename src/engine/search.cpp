#include "engine/search.h"
#include "engine/evaluate.h"
#include "chess/movegen.h"
#include <vector>

namespace chess::engine {
namespace {

constexpr int mate_score = 100000;
constexpr int INF = 1000000;

}

int negamax(const Board& board, int depth){
    if (depth == 0) return evaluate(board);

    std::vector<Move> moves = generate_legal_moves(board);

    if (moves.empty()){
        if (in_check(board, board.side_to_move())){
            return -mate_score - depth;
        } 
        return 0;

    }

    int best_score = -INF;

    for (const Move& move : moves) {
        Board next = board;
        next.make_move(move);

        int score = -negamax(next, depth - 1);
        
        if (score > best_score){
            best_score = score;
        }

    }

    return best_score;

}

SearchResult search_best_move(const Board& board, int depth){

    SearchResult result;
    result.score = -INF;

    std::vector<Move> moves = generate_legal_moves(board);

    if (moves.empty()){
        result.score = negamax(board, depth);
        return result;
    }    
    
    for (const Move& move : moves) {
        
        Board next = board;
        next.make_move(move);

        int score = -negamax(next, depth - 1);

        if (score > result.score){
            result.score = score;
            result.move = move;

        }
    }

    return result;

}


}

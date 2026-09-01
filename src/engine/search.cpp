#include "engine/search.h"
#include "engine/evaluate.h"
#include "chess/movegen.h"
#include <vector>

namespace chess::engine {
namespace {

constexpr int mate_score = 100000;
constexpr int INF = 1000000;

}

int Searcher::negamax(Board& board, int depth, int alpha, int beta){
    stats_.nodes++;
    if (depth == 0) return evaluate(board, model_);

    std::vector<Move> moves = generate_legal_moves(board);

    if (moves.empty()){
        if (in_check(board, board.side_to_move())){
            return -mate_score - depth;
        } 
        return 0;

    }

    int best_score = -INF;

    for (const Move& move : moves) {
        UndoState undo_state = board.make_move(move);

        int score = -negamax(board, depth - 1, -beta, -alpha);
        
        board.unmake_move(move, undo_state);

        if (score > best_score){
            best_score = score;
        }

        if (score > alpha){
            alpha = score;
        }

        if (alpha >= beta){
            stats_.beta_cutoffs++;
            break;
        }

    }

    return best_score;

}

SearchResult Searcher::search_best_move(Board& board, int depth){

    SearchResult result;
    result.score = -INF;

    std::vector<Move> moves = generate_legal_moves(board);

    if (moves.empty()){
        result.score = negamax(board, depth, -INF, INF);
        return result;
    }    
    
    int alpha = -INF;
    int beta = INF;

    for (const Move& move : moves) {
        UndoState undo_state = board.make_move(move);
        
        int score = -negamax(board, depth - 1, -beta, -alpha);

        board.unmake_move(move, undo_state);

        if (score > result.score){
            result.score = score;
            result.move = move;
        }

        if (score > alpha) {
            alpha = score;
        }


    }

    return result;

}

void Searcher::resetStats(){
    stats_ = SearchStats{};
}

}

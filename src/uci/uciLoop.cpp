#include "uci/uciLoop.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "chess/movegen.h"

const std::string ENGINE_NAME = "chess-bot";
const std::string AUTHOR_NAME = "Nate Lamarche";

namespace {

constexpr int default_search_depth = 5;

char promotion_to_char(chess::Piece piece) {
    switch (piece) {
        case chess::Piece::Knight: return 'n';
        case chess::Piece::Bishop: return 'b';
        case chess::Piece::Rook: return 'r';
        case chess::Piece::Queen: return 'q';
        default: return '\0';
    }
}

chess::Piece promotion_from_char(char piece) {
    switch (piece) {
        case 'n': return chess::Piece::Knight;
        case 'b': return chess::Piece::Bishop;
        case 'r': return chess::Piece::Rook;
        case 'q': return chess::Piece::Queen;
        default: return chess::Piece::None;
    }
}

bool is_square_text(const std::string& text, std::size_t offset) {
    return text[offset] >= 'a' && text[offset] <= 'h'
        && text[offset + 1] >= '1' && text[offset + 1] <= '8';
}

chess::Square square_from_text(const std::string& text, std::size_t offset) {
    return static_cast<chess::Square>(
        (text[offset] - 'a') + 8 * (text[offset + 1] - '1'));
}

}

void UciLoop::run() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input{line};
        std::string command;
        input >> command;

        if (command == "quit") {
            break;
        }

        handle_command(line);
    }
}

void UciLoop::handle_command(const std::string& line) {
    std::istringstream input{line};
    std::string command;
    input >> command;

    if (command == "uci") {
        std::cout << "id name " + ENGINE_NAME + "\n"
                  << "id author " + AUTHOR_NAME + "\n"
                  << "uciok" << std::endl;
    } else if (command == "isready") {
        std::cout << "readyok" << std::endl;
    } else if (command == "ucinewgame") {
        board_.set_start_position();
    } else if (command == "position") {
        handle_position(input);
    } else if (command == "go") {
        handle_go(input);
    }
}

void UciLoop::handle_position(std::istringstream& input) {
    std::string token;
    if (!(input >> token)) {
        return;
    }

    chess::Board next;

    try {
        if (token == "startpos") {
            next.set_start_position();
            token.clear();
            input >> token;
        } else if (token == "fen") {
            std::string fen;
            for (int field = 0; field < 6; field++) {
                std::string part;
                if (!(input >> part)) {
                    return;
                }
                if (!fen.empty()) {
                    fen += ' ';
                }
                fen += part;
            }
            next.set_fen(fen);
            token.clear();
            input >> token;
        } else {
            return;
        }
    } catch (const std::runtime_error&) {
        return;
    }

    if (token.empty()) {
        board_ = next;
        return;
    }

    if (token != "moves") {
        return;
    }

    while (input >> token) {
        const auto move = parse_move(next, token);
        if (!move) {
            return;
        }
        next.make_move(*move);
    }

    board_ = next;
}

void UciLoop::handle_go(std::istringstream& input) {
    int depth = default_search_depth;
    std::string token;

    while (input >> token) {
        if (token != "depth") {
            continue;
        }

        int requested_depth = 0;
        if (input >> requested_depth && requested_depth > 0) {
            depth = requested_depth;
        }
    }

    const std::vector<chess::Move> legal_moves =
        chess::generate_legal_moves(board_);
    if (legal_moves.empty()) {
        std::cout << "bestmove 0000" << std::endl;
        return;
    }

    const chess::engine::SearchResult result =
        chess::engine::search_best_move(board_, depth);

    std::cout << "info depth " << depth << " score cp " << result.score << '\n'
              << "bestmove " << move_to_uci(result.move) << std::endl;
}

std::string UciLoop::move_to_uci(const chess::Move& move) {
    std::string text;
    text += static_cast<char>('a' + move.from % 8);
    text += static_cast<char>('1' + move.from / 8);
    text += static_cast<char>('a' + move.to % 8);
    text += static_cast<char>('1' + move.to / 8);

    const char promotion = promotion_to_char(move.promotion);
    if (promotion != '\0') {
        text += promotion;
    }

    return text;
}

std::optional<chess::Move>
UciLoop::parse_move(const chess::Board& board, const std::string& text) {
    if ((text.size() != 4 && text.size() != 5)
        || !is_square_text(text, 0)
        || !is_square_text(text, 2)) {
        return std::nullopt;
    }

    const chess::Square from = square_from_text(text, 0);
    const chess::Square to = square_from_text(text, 2);
    chess::Piece promotion = chess::Piece::None;

    if (text.size() == 5) {
        promotion = promotion_from_char(text[4]);
        if (promotion == chess::Piece::None) {
            return std::nullopt;
        }
    }

    const std::vector<chess::Move> legal_moves =
        chess::generate_legal_moves(board);
    const auto match = std::find_if(
        legal_moves.begin(), legal_moves.end(),
        [from, to, promotion](const chess::Move& move) {
            return move.from == from
                && move.to == to
                && move.promotion == promotion;
        });

    if (match == legal_moves.end()) {
        return std::nullopt;
    }

    return *match;
}

int main() {
    UciLoop uci;
    uci.run();
    return 0;
}

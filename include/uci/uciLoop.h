#pragma once

#include <string>
#include <sstream>
#include <optional>

#include "engine/search.h"
#include "chess/board.h"

class UciLoop {

public:
    void run();

private:
    chess::Board board_;

    void handle_command(const std::string& line);
    void handle_position(std::istringstream& input);
    void handle_go(std::istringstream& input);

    static std::string move_to_uci(const chess::Move& move);
    static std::optional<chess::Move>
        parse_move(chess::Board& board, const std::string& text);
};
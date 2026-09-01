#pragma once

#include <string>
#include <sstream>
#include <optional>

#include "chess/board.h"
#include "engine/nnue_model.h"

class UciLoop {

public:
    UciLoop();
    void run();

private:
    chess::engine::NnueModel model_;
    chess::Board board_;

    void handle_command(const std::string& line);
    void handle_position(std::istringstream& input);
    void handle_go(std::istringstream& input);

    static std::string move_to_uci(const chess::Move& move);
    static std::optional<chess::Move>
        parse_move(chess::Board& board, const std::string& text);
};
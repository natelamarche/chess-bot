cmake_minimum_required(VERSION 3.16)
project(chess_bot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library: core engine
add_library(chess
    src/chess/board.cpp
    src/chess/move.cpp
    src/chess/movegen.cpp
    src/perft/perft.cpp
    src/engine/evaluate.cpp
    src/engine/search.cpp
)

target_include_directories(chess
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(chess PRIVATE
    -Wall -Wextra -Wpedantic
)

# App: simple CLI
add_executable(chess_cli apps/main.cpp)
target_link_libraries(chess_cli PRIVATE chess)

# Tests: perft
add_executable(perft_test tests/perft/perft_tests.cpp)
target_link_libraries(perft_test PRIVATE chess)

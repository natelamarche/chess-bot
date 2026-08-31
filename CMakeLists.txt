cmake_minimum_required(VERSION 3.16)
project(chess_bot LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library: core engine
add_library(chess
    src/chess/board.cpp
    src/chess/move.cpp
    src/chess/movegen.cpp
    src/perft/perft.cpp
    src/engine/evaluate.cpp
    src/engine/search.cpp
    src/engine/nnue_model.cpp
    src/engine/nnue_features.cpp
    src/engine/nnue_state.cpp
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

# Tests: NNUE feature encoding
add_executable(nnue_features_test tests/nnue/nnue_features_tests.cpp)
target_link_libraries(nnue_features_test PRIVATE chess)
add_test(NAME nnue_features_test COMMAND nnue_features_test)

# Tests: NNUE accumulator rebuilding
add_executable(nnue_state_test tests/nnue/nnue_state_tests.cpp)
target_link_libraries(nnue_state_test PRIVATE chess)
add_test(NAME nnue_state_test COMMAND nnue_state_test)
set_tests_properties(nnue_state_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Tests: search benchmark
add_executable(search_bench tests/search-benchmark/search_bench.cpp)
target_link_libraries(search_bench PRIVATE chess)

# UCI
add_executable(engine src/uci/uciLoop.cpp)
target_link_libraries(engine PRIVATE chess)

install(TARGETS engine DESTINATION bin)

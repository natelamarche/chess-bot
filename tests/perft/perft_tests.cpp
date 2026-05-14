#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include "perft/perft.h"
#include "chess/board.h"

struct PerftCase {
    std::string fen;
    std::map<int, std::uint64_t> expected_by_depth;
};

using namespace std;
int main(int argc, char* argv[]) {
    int depth = 4;
    if (argc > 1){
        depth = stoi(argv[1]);
    }

    if (depth < 1 || depth > 6) {
        cout << "Improper depth value (1 <= depth <= 6)\n";
        return 1;
    }

    string file_path = "tests/perft/classical.txt";
    if (argc > 2){
        file_path = argv[2];
    }

    fstream file{file_path};
    if (!file.is_open()){
        cout << "Could not open file: " << file_path << "\n";
        return 1;
    }

    string line;

    int caseNum = 1;
    int testsRun = 0;
    int testsSkipped = 0;
    while (getline(file, line)){
        if (line.empty()) {
            caseNum++;
            continue;
        }

        PerftCase position;
        stringstream ss{line};
        string part;

        getline(ss, part, ';');
        position.fen = part;

        while (getline(ss, part, ';')){
            string depth_token;
            uint64_t expected = 0;
            stringstream part_stream{part};

            if (!(part_stream >> depth_token >> expected) ||
                depth_token.size() < 2 ||
                depth_token[0] != 'D') {
                cout << "ERROR: Could not parse perft depth on line " << caseNum
                    << "\n Token: " << part
                    << "\n";
                return 1;
            }

            int parsed_depth = stoi(depth_token.substr(1));
            position.expected_by_depth[parsed_depth] = expected;

        }

        auto expected = position.expected_by_depth.find(depth);
        if (expected == position.expected_by_depth.end()){
            testsSkipped++;
            caseNum++;
            continue;
        }

        chess::Board board;
        board.set_fen(position.fen);
        uint64_t result = chess::perft(board, depth);

        if (result != expected->second){
            cout << "ERROR: Line number: " << caseNum 
                << "\n FEN: " << position.fen 
                << "\n Result: " << result
                << "\n Expected: " << expected->second
                << "\n";
                return 1;
        } 

        testsRun++;
        caseNum++;
    }

    if (testsRun == 0) {
        cout << "No test cases found for depth " << depth << ".\n";
        return 1;
    }

    cout << "All tests passed! Ran " << testsRun
        << " cases at depth " << depth
        << " and skipped " << testsSkipped
        << " cases without that depth.\n";

    return 0;

}

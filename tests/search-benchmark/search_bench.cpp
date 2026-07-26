#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

#include "chess/board.h"
#include "engine/search.h"

using namespace std;
int main(int argc, char* argv[]) {
    int depth = 4;
    if (argc > 1){
        depth = stoi(argv[1]);
    }

    if (depth < 1 || depth > 8) {
        cout << "Improper depth value (1 <= depth <= 8)\n";
        return 1;
    }

    string file_path = "tests/classical.txt";
    
    fstream file{file_path};
    if (!file.is_open()){
        cout << "Could not open file: " << file_path << "\n";
        return 1;
    }

    streambuf* oldCoutBuf;
    ofstream outFile;

    if (argc > 2){
        outFile.open(argv[2], ios::app);
        
        if (!outFile){
            cout << "Could not open output file: " << argv[2] << "\n";
            return 1;
        }


        oldCoutBuf = cout.rdbuf(outFile.rdbuf());
    }

    string line;
    chess::engine::Searcher searcher{};
    
    const chrono::steady_clock::time_point  start = chrono::steady_clock::now();
    int testsRun = 0;
    
    while (getline(file, line)){
        if (line.empty()) {
            continue;
        }

        stringstream ss{line};
        string part;

        getline(ss, part, ';');
        string fen = part;

        chess::Board board;
        board.set_fen(fen);

        chess::engine::SearchResult result = searcher.search_best_move(board, depth);

        testsRun++;
    }

    if (testsRun == 0) {
        cout << "No test cases run.\n";
        return 1;
    }

    chess::engine::SearchStats stats = searcher.getStats();

    const chrono::steady_clock::time_point end = chrono::steady_clock::now();
    const chrono::milliseconds elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "All cases run! Ran " << testsRun
        << " positions at depth " << depth << "\n"
        << "    Nodes: " << stats.nodes << "\n"
        << "    Quiescence nodes: " << stats.quiescence_nodes << "\n"
        << "    Beta cutoffs: " << stats.beta_cutoffs << "\n"
        << "    Time: " << elapsed.count() << "ms\n"
        << "    NPS: " << stats.nodes/elapsed.count()*1000 << endl;

    if (oldCoutBuf){
        cout.rdbuf(oldCoutBuf);
    }

    return 0;

}

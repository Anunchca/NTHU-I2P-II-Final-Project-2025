#pragma once
#include <cstdint>
#include <vector>

enum TTFlag { TT_EXACT, TT_LOWER, TT_UPPER };

struct TTEntry {
    uint64_t key = 0;
    int depth = -1;
    int score = 0;
    TTFlag flag = TT_EXACT;
    Move best_move;
    bool valid = false;
};

class TranspositionTable {
public:
    TranspositionTable(size_t size_mb = 64) {
        size_t num_entries = (size_mb * 1024 * 1024) / sizeof(TTEntry);
        table.resize(num_entries);
        mask = num_entries - 1;
    }

    void clear() {
        std::fill(table.begin(), table.end(), TTEntry{});
    }

    TTEntry* probe(uint64_t key) {
        TTEntry& e = table[key & mask];
        if (e.valid && e.key == key) return &e;
        return nullptr;
    }

    void store(uint64_t key, int depth, int score, TTFlag flag, const Move& move) {
        TTEntry& e = table[key & mask];
        if (!e.valid || depth >= e.depth) {
            e.key = key;
            e.depth = depth;
            e.score = score;
            e.flag = flag;
            e.best_move = move;
            e.valid = true;
        }
    }

private:
    std::vector<TTEntry> table;
    size_t mask;
};
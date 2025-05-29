#ifndef RUN_PARAMS_HPP
#define RUN_PARAMS_HPP

#include <string>
#include <cstddef>
#include <iostream>

namespace cellib::run {

struct run_params {
    std::string automaton = "game-of-life";

    std::string device = "CPU";
    std::string traverser = "standard";
    std::string evaluator = "standard";
    std::string layout = "standard";

    std::string reference_impl = "none";
    
    int x_size = 0;
    int y_size = 0;
    int steps = 0;

    int precision = 0;

    int x_tile_size = 0;
    int y_tile_size = 0;

    int rounds = 1;
    int warmup_rounds = 0;

    int seed = 42;

    bool print = false;
    bool help = false;

    bool print_csv_header = false;

    int cuda_block_size_x = 32;
    int cuda_block_size_y = 8;

    void print_to(std::ostream& os) {
        os << "Run Parameters:\n";
        os << "  Automaton: " << automaton << "\n";
        os << "  Device: " << device << "\n";
        os << "  Traverser: " << traverser << "\n";
        os << "  Evaluator: " << evaluator << "\n";
        os << "  Layout: " << layout << "\n";
        os << "  Reference Implementation: " << reference_impl << "\n";
        os << "  Grid Size: (" << x_size << ", " << y_size << ")\n";
        os << "  Steps: " << steps << "\n";
        os << "  Print: " << (print ? "true" : "false") << "\n";
        os << "  Precision: " << precision << "\n";
        os << "  Rounds: " << rounds << "\n";
        os << "  Warmup Rounds: " << warmup_rounds << "\n";
        os << "  Seed: " << seed << "\n";
        os << "  X Tile Size: " << x_tile_size << "\n";
        os << "  Y Tile Size: " << y_tile_size << "\n";
        os << "  CUDA Block Size: (" << cuda_block_size_x << ", " << cuda_block_size_y << ")\n";
        os << "  Print CSV Header: " << (print_csv_header ? "true" : "false") << "\n";
        os << "  Help: " << (help ? "true" : "false") << "\n";
    }

    void print_std() {
        print_to(std::cout);
    }

    static std::string csv_header() {
        return "automaton,device,traverser,evaluator,layout,reference_impl,x_size,y_size,steps,rounds,warmup_rounds,x_tile_size,y_tile_size,cuda_block_size_x,cuda_block_size_y,seed,precision";
    }

    std::string csv_line() const {
        return automaton + "," +
               device + "," +
               traverser + "," +
               evaluator + "," +
               layout + "," +
               (reference_impl != "none" ? reference_impl : "") + "," +
               std::to_string(x_size) + "," +
               std::to_string(y_size) + "," +
               std::to_string(steps) + "," +
               std::to_string(rounds) + "," +
               std::to_string(warmup_rounds) + "," +
               std::to_string(x_tile_size) + "," +
               std::to_string(y_tile_size) + "," +
               std::to_string(cuda_block_size_x) + "," +
               std::to_string(cuda_block_size_y) + "," +
               std::to_string(seed) + "," +
               std::to_string(precision);
    }
};

}

#endif // RUN_PARAMS_HPP
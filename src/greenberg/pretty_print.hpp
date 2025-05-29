#ifndef GREENBERG_HASTINGS_PRETTY_PRINT_HPP
#define GREENBERG_HASTINGS_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"

namespace greenberg {

using print_config = cellib::memory::grids::standard::print_config<ghm_cell_state>;

struct ghm_pretty_print {
    static print_config get_config() {
        return print_config()
            .with(ghm_cell_state::quiescent, "\033[90m·\033[0m")       // Dark gray dot for quiescent
            .with(ghm_cell_state::excited, "\033[1;31m█\033[0m")       // Bright red for excited
            .with(ghm_cell_state::refractory_1, "\033[1;35m█\033[0m")  // Magenta
            .with(ghm_cell_state::refractory_2, "\033[1;34m█\033[0m")  // Blue
            .with(ghm_cell_state::refractory_3, "\033[1;36m█\033[0m")  // Cyan
            .with(ghm_cell_state::refractory_4, "\033[1;32m█\033[0m")  // Green
            .with(ghm_cell_state::refractory_5, "\033[1;33m█\033[0m")  // Yellow
            .with(ghm_cell_state::refractory_6, "\033[38;5;208m█\033[0m"); // Orange
    }
};

}

#endif // GREENBERG_HASTINGS_PRETTY_PRINT_HPP

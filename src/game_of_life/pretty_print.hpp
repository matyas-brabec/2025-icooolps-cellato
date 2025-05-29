#ifndef CELLATO_PRETTY_PRINT_HPP
#define CELLATO_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"

namespace game_of_life {

using print_config = cellato::memory::grids::standard::print_config<gol_cell_state>;

struct gol_pretty_print {
    static print_config get_config() {
        return print_config()
            .with(gol_cell_state::dead, "\033[1;31m.\033[0m")
            .with(gol_cell_state::alive, "\033[1;32m#\033[0m");
    }
};

} // namespace game_of_life

#endif // CELLATO_PRETTY_PRINT_HPP
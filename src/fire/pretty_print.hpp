#ifndef FOREST_FIRE_PRETTY_PRINT_HPP
#define FOREST_FIRE_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"

namespace fire {

using print_config = cellib::memory::grids::standard::print_config<fire_cell_state>;

struct fire_pretty_print {
    static print_config get_config() {
        return print_config()
            .with(fire_cell_state::empty, "\033[90m.\033[0m")  // Dark grey for empty ground (almost invisible)
            .with(fire_cell_state::tree, "\033[1;32m#\033[0m") // Bright green for trees
            .with(fire_cell_state::ash, "\033[1;37m*\033[0m")  // Light gray for ash
            .with(fire_cell_state::fire, "\033[1;31m@\033[0m"); // Bright red for fire
    }
};

}

#endif // FOREST_FIRE_PRETTY_PRINT_HPP

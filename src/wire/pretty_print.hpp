#ifndef WIRE_PRETTY_PRINT_HPP
#define WIRE_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"

namespace wire {

using print_config = cellato::memory::grids::standard::print_config<wire_cell_state>;

struct wire_pretty_print {
    static print_config get_config() {
        return print_config()
            .with(wire_cell_state::empty, "\033[1;30m.\033[0m")      // Black for empty
            .with(wire_cell_state::electron_head, "\033[1;34m@\033[0m") // Blue for electron head
            .with(wire_cell_state::electron_tail, "\033[1;31m#\033[0m") // Red for electron tail
            .with(wire_cell_state::conductor, "\033[1;33m-\033[0m");   // Yellow for conductor
    }
};

}

#endif // WIRE_PRETTY_PRINT_HPP

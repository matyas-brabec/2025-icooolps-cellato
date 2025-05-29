#ifndef FOREST_FIRE_CONFIG_HPP
#define FOREST_FIRE_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

#include <string>
namespace fire {

struct config {

    static constexpr auto name = "forest-fire";

    using algorithm = fire_algorithm;
    
    using cell_state = fire_cell_state;
    using state_dictionary = cellato::memory::grids::state_dictionary<
        cell_state::empty, cell_state::tree,
        cell_state::fire, cell_state::ash>;

    using pretty_print = fire_pretty_print;

    using reference_implementation = reference::runner;

    struct input {
        using random = fire_random_init;
    };
};

}

#endif // FOREST_FIRE_CONFIG_HPP

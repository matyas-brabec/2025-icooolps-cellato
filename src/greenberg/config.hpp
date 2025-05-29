#ifndef GREENBERG_HASTINGS_CONFIG_HPP
#define GREENBERG_HASTINGS_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

namespace greenberg {

struct config {
    
    static constexpr auto name = "greenberg-hastings";

    using algorithm = ghm_algorithm;

    using cell_state = ghm_cell_state;
    using state_dictionary = cellib::memory::grids::state_dictionary<
        ghm_cell_state::quiescent, ghm_cell_state::excited,
        ghm_cell_state::refractory_1, ghm_cell_state::refractory_2,
        ghm_cell_state::refractory_3, ghm_cell_state::refractory_4,
        ghm_cell_state::refractory_5, ghm_cell_state::refractory_6>;

    using reference_implementation = reference::runner;
    
    using pretty_print = ghm_pretty_print;
    
    struct input {
        using random = ghm_random_init;
    };
};

}

#endif // GREENBERG_HASTINGS_CONFIG_HPP

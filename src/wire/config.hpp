#ifndef WIRE_CONFIG_HPP
#define WIRE_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

namespace wire {

struct config {

    static constexpr auto name = "wire";

    using algorithm = wire_algorithm;
    
    using cell_state = wire_cell_state;
    using state_dictionary = cellato::memory::grids::state_dictionary<
        cell_state::empty, cell_state::electron_head,
        cell_state::electron_tail, cell_state::conductor>;

    using pretty_print = wire_pretty_print;

    using reference_implementation = reference::runner;

    struct input {
        using random = wire_random_init;
    };
};

}

#endif // WIRE_CONFIG_HPP

#ifndef GAME_OF_LIFE_CONFIG_HPP
#define GAME_OF_LIFE_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

namespace game_of_life {

struct config {

    static constexpr auto name = "game-of-life";

    using algorithm = gol_algorithm;
    
    using cell_state = gol_cell_state;
    using state_dictionary = cellato::memory::grids::state_dictionary<
        cell_state::dead, cell_state::alive>;

    using pretty_print = gol_pretty_print;

    using reference_implementation = reference::runner;

    struct input {
        using random = gol_random_init;
    };
};

} // namespace game_of_life

#endif // GAME_OF_LIFE_CONFIG_HPP
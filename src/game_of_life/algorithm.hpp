#ifndef GAME_OF_LIFE_ALGORITHM_HPP
#define GAME_OF_LIFE_ALGORITHM_HPP

#include "core/ast.hpp"

namespace game_of_life {
using namespace cellato::ast;

enum class gol_cell_state {
    dead,
    alive
};

// Define constants for cell states
using alive = state_constant<gol_cell_state::alive>;
using dead = state_constant<gol_cell_state::dead>;

// Define integer constants
using c_2 = constant<2>;
using c_3 = constant<3>;

// Define predicates for cell state checks
using cell_is_alive = p<current_state, equals, alive>;
using cell_is_dead = p<current_state, equals, dead>;

// Count neighbors in Moore neighborhood
using alive_count = count_neighbors<alive, moore_8_neighbors>;

// Define predicates for neighbor count checks
using has_two_alive_neighbors = p<alive_count, equals, c_2>;
using has_three_alive_neighbors = p<alive_count, equals, c_3>;
using has_two_or_three_alive_neighbors = p<has_two_alive_neighbors, or_, has_three_alive_neighbors>;

// Define the Game of Life algorithm
using gol_algorithm = 
    if_< cell_is_alive >::then_<
        if_< has_two_or_three_alive_neighbors >::then_<
            alive
        >::else_<
            dead
        >
    >::
    else_< // cell_is_dead
        if_< has_three_alive_neighbors >::then_<
            alive
        >::else_<
            dead
        >
    >;

}

#endif // GAME_OF_LIFE_ALGORITHM_HPP
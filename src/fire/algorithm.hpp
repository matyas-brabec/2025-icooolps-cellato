#ifndef FOREST_FIRE_ALGORITHM_HPP
#define FOREST_FIRE_ALGORITHM_HPP

#include "core/ast.hpp"

namespace fire {
using namespace cellib::ast;

enum class fire_cell_state {
    empty,
    tree,
    ash,
    fire
};

// Define constants for cell states
using empty = state_constant<fire_cell_state::empty>;
using tree = state_constant<fire_cell_state::tree>;
using ash = state_constant<fire_cell_state::ash>;
using fire = state_constant<fire_cell_state::fire>;

// Define integer constants
using c_0 = constant<0>;

// Define predicates for cell state checks
using cell_is_empty = p<current_state, equals, empty>;
using cell_is_tree = p<current_state, equals, tree>;
using cell_is_ash = p<current_state, equals, ash>;
using cell_is_fire = p<current_state, equals, fire>;

// Count fire cells in the von Neumann neighborhood
using fire_count = count_neighbors<fire, von_neumann_4_neighbors>;

// Define predicates for neighbor checks
using has_fire_neighbors = p<fire_count, greater_than, c_0>;
using no_fire_neighbors = p<fire_count, equals, c_0>;

// Define the Forest Fire algorithm
using fire_algorithm = 
    if_< cell_is_fire >::then_<
        ash
    >::
    elif_< cell_is_ash >::then_<
        if_< has_fire_neighbors >::then_<
            ash
        >::else_<
            empty
        >
    >::
    elif_< cell_is_tree >::then_<
        if_< has_fire_neighbors >::then_<
            fire
        >::else_<
            tree
        >
    >::
    else_< empty >; // If cell is empty, it remains empty

}

#endif // FOREST_FIRE_ALGORITHM_HPP

#ifndef GREENBERG_HASTINGS_ALGORITHM_HPP
#define GREENBERG_HASTINGS_ALGORITHM_HPP

#include "core/ast.hpp"

namespace greenberg {
using namespace cellato::ast;

// Define 8 states: 0 (quiescent), 1 (excited), 2-7 (refractory)
enum class ghm_cell_state {
    quiescent = 0,
    excited = 1,
    refractory_1 = 2,
    refractory_2 = 3,
    refractory_3 = 4,
    refractory_4 = 5,
    refractory_5 = 6,
    refractory_6 = 7
};

// Constants for each state
using quiescent = state_constant<ghm_cell_state::quiescent>;
using excited = state_constant<ghm_cell_state::excited>;
using refractory_1 = state_constant<ghm_cell_state::refractory_1>;
using refractory_2 = state_constant<ghm_cell_state::refractory_2>;
using refractory_3 = state_constant<ghm_cell_state::refractory_3>;
using refractory_4 = state_constant<ghm_cell_state::refractory_4>;
using refractory_5 = state_constant<ghm_cell_state::refractory_5>;
using refractory_6 = state_constant<ghm_cell_state::refractory_6>;

// Define integer constants
using c_0 = constant<0>;

// Define predicates for cell state checks
using cell_is_quiescent = p<current_state, equals, quiescent>;
using cell_is_excited = p<current_state, equals, excited>;
using cell_is_refractory_1 = p<current_state, equals, refractory_1>;
using cell_is_refractory_2 = p<current_state, equals, refractory_2>;
using cell_is_refractory_3 = p<current_state, equals, refractory_3>;
using cell_is_refractory_4 = p<current_state, equals, refractory_4>;
using cell_is_refractory_5 = p<current_state, equals, refractory_5>;
using cell_is_refractory_6 = p<current_state, equals, refractory_6>;

// Count excited neighbors in Moore neighborhood
using excited_count = count_neighbors<excited, moore_8_neighbors>;

// Check if any neighbor is excited
using has_excited_neighbors = p<excited_count, greater_than, c_0>;

// Define the Greenberg-Hastings Model algorithm
/*
Rules:
1. If cell is quiescent (0) and has at least one excited neighbor (1), it becomes excited (1)
2. If cell is excited (1), it transitions to refractory_1 (2)
3. If cell is in refractory_n state, it transitions to refractory_n+1 or back to quiescent
*/
using ghm_algorithm = 
    if_< cell_is_quiescent >::then_<
        if_< has_excited_neighbors >::then_<
            excited
        >::else_<
            quiescent
        >
    >::
    elif_< cell_is_excited      >::then_< refractory_1 >::
    elif_< cell_is_refractory_1 >::then_< refractory_2 >::
    elif_< cell_is_refractory_2 >::then_< refractory_3 >::
    elif_< cell_is_refractory_3 >::then_< refractory_4 >::
    elif_< cell_is_refractory_4 >::then_< refractory_5 >::
    elif_< cell_is_refractory_5 >::then_< refractory_6 >::
    else_< quiescent >; // refractory_6 goes back to quiescent

}

#endif // GREENBERG_HASTINGS_ALGORITHM_HPP

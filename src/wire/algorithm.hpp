#ifndef WIRE_ALGORITHM_HPP
#define WIRE_ALGORITHM_HPP

#include "core/ast.hpp"

namespace wire {
using namespace cellato::ast;

// --- States for the Wireworld CA ---
enum class wire_cell_state { 
    empty, 
    electron_head, 
    electron_tail, 
    conductor 
};

// --- Define constants for cell states ---
using empty = state_constant<wire_cell_state::empty>;
using electron_head = state_constant<wire_cell_state::electron_head>;
using electron_tail = state_constant<wire_cell_state::electron_tail>;
using conductor = state_constant<wire_cell_state::conductor>;

// --- Integer constants ---
using c_1 = constant<1>;
using c_2 = constant<2>;

// --- Predicates for cell state checks ---
using cell_is_empty = p<current_state, equals, empty>;
using cell_is_electron_head = p<current_state, equals, electron_head>;
using cell_is_electron_tail = p<current_state, equals, electron_tail>;
using cell_is_conductor = p<current_state, equals, conductor>;

// --- Count electron heads in the Moore neighborhood ---
using electron_head_count = count_neighbors<electron_head, moore_8_neighbors>;

// --- Check if exactly 1 or 2 neighboring cells are electron heads ---
using has_one_electron_head_neighbor = p<electron_head_count, equals, c_1>;
using has_two_electron_head_neighbors = p<electron_head_count, equals, c_2>;
using has_one_or_two_electron_head_neighbors = 
    p<has_one_electron_head_neighbor, or_, has_two_electron_head_neighbors>;

// --- Wireworld algorithm ---
// Rule 1: empty → empty
// Rule 2: electron head → electron tail
// Rule 3: electron tail → conductor
// Rule 4: conductor → electron head if exactly 1 or 2 neighboring cells 
//         are electron heads, otherwise remains conductor
using wire_algorithm = 
    if_< cell_is_electron_head >::then_<
        electron_tail
    >::elif_< cell_is_electron_tail >::then_<
        conductor
    >::elif_< cell_is_conductor >::then_<
        if_<has_one_or_two_electron_head_neighbors>::then_<
            electron_head
        >::else_<
            conductor
        >
    >::else_<empty>; // If cell is empty, it remains empty
}

#endif // WIRE_ALGORITHM_HPP

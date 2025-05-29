#ifndef CELLIB_BIT_ARRAY_EVALUATORS_HPP
#define CELLIB_BIT_ARRAY_EVALUATORS_HPP

#include "../core/ast.hpp"
#include "../memory/interface.hpp"
#include <cstddef>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace cellib::evaluators::bit_array {

using namespace cellib::ast;
using namespace cellib::memory;

template <typename grid_t>
using state_t = grids::point_in_grid<typename grid_t::cell_ptr_t>;

template <std::size_t to>
struct static_for {
    template <typename Func>
    CUDA_CALLABLE static void apply(Func func) {
        if constexpr (to > 0) {
            func.template operator()<to - 1>();
            static_for<to - 1>::apply(func);
        }
    }
};

// Implementation evaluator - processes a single subcell
template <typename grid_t, typename Expression, std::size_t subcell_offset>
struct _impl_evaluator;

// Main evaluator - processes an entire word at once
template <typename bit_array_grid_t, typename Expression>
struct evaluator {
    using store_word_type = typename bit_array_grid_t::store_type;
    static constexpr int cells_per_word = bit_array_grid_t::cells_per_word;

    CUDA_CALLABLE static store_word_type evaluate(state_t<bit_array_grid_t> state) {
        // Create a new word to store the results
        store_word_type result_word = 0;
        (void)state;
        
        // Iterate over each subcell and evaluate the expression
        static_for<cells_per_word>::apply([&]<std::size_t subcell_idx>() {
            // Calculate and evaluate each subcell
            auto cell_result = _impl_evaluator<bit_array_grid_t, Expression, subcell_idx>::evaluate(state);
            // #ifndef __CUDACC__
            // std::cout << "Subcell " << subcell_idx << ": " << cell_result << std::endl;
            // #endif
            
            // Position this subcell in the result word
            result_word |= cell_result << (subcell_idx * bit_array_grid_t::bits_per_cell);
        });

        // #ifndef __CUDACC__
        // exit(0);
        // #endif
        
        return result_word;
    }
};


// Implement specific expression evaluators below

// Constants
template <typename grid_t, auto Value, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, constant<Value>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> /* state */) {
        return Value;
    }
};

// State constants
template <typename grid_t, typename state_type, state_type Value, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, state_constant<Value>, subcell_offset> {
    using store_type = typename grid_t::store_type;
    using dictionary_t = typename grid_t::states_dict_t;

    CUDA_CALLABLE static store_type evaluate(state_t<grid_t> /* state */) {
        constexpr store_type index = dictionary_t::state_to_index(Value);
        return index;
    }
};

// Conditional evaluation
template <typename grid_t, typename Condition, typename Then, typename Else, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, if_then_else<Condition, Then, Else>, subcell_offset> {

    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        if (_impl_evaluator<grid_t, Condition, subcell_offset>::evaluate(state)) {
            return _impl_evaluator<grid_t, Then, subcell_offset>::evaluate(state);
        } else {
            return _impl_evaluator<grid_t, Else, subcell_offset>::evaluate(state);
        }
    }
};

// Logical operators
template <typename grid_t, typename Left, typename Right, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, and_<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static bool evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) && 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, or_<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static bool evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) || 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

// Comparison operators
template <typename grid_t, typename Left, typename Right, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, equals<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static bool evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) == 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, not_equals<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static bool evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) != 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, greater_than<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static bool evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) > 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

// Neighborhood access
template <typename grid_t, int x_offset, int y_offset, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, neighbor_at<x_offset, y_offset>, subcell_offset> {
    using store_type = typename grid_t::store_type;
    constexpr static auto cells_per_word = grid_t::cells_per_word;

    CUDA_CALLABLE static store_type evaluate(state_t<grid_t> state) {
        std::size_t x_size_original = state.properties.x_size * cells_per_word;

        std::size_t x_original = state.position.x * cells_per_word + subcell_offset + static_cast<std::size_t>(x_offset);
        std::size_t y_original = state.position.y + static_cast<std::size_t>(y_offset);

        return state.grid.get_individual_cell_at(y_original * x_size_original + x_original);
    }
};

// Neighbor counting
template <typename grid_t, typename CellStateValue, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, count_neighbors<CellStateValue, moore_8_neighbors>, subcell_offset> {

    template <int x_offset, int y_offset>
    using cell_at = _impl_evaluator<grid_t, neighbor_at<x_offset, y_offset>, subcell_offset>;

    CUDA_CALLABLE static int evaluate(state_t<grid_t> state) {
        auto target_state = _impl_evaluator<grid_t, CellStateValue, subcell_offset>::evaluate(state);

        auto top_left_c     = cell_at<-1, -1>::evaluate(state) == target_state ? 1 : 0;
        auto top_c          = cell_at< 0, -1>::evaluate(state) == target_state ? 1 : 0;
        auto top_right_c    = cell_at< 1, -1>::evaluate(state) == target_state ? 1 : 0;
        auto left_c         = cell_at<-1,  0>::evaluate(state) == target_state ? 1 : 0;
        auto right_c        = cell_at< 1,  0>::evaluate(state) == target_state ? 1 : 0;
        auto bottom_left_c  = cell_at<-1,  1>::evaluate(state) == target_state ? 1 : 0;
        auto bottom_c       = cell_at< 0,  1>::evaluate(state) == target_state ? 1 : 0;
        auto bottom_right_c = cell_at< 1,  1>::evaluate(state) == target_state ? 1 : 0;

        return top_left_c + top_c + top_right_c +
               left_c + right_c +
               bottom_left_c + bottom_c + bottom_right_c;
    }
};

template <typename grid_t, typename CellStateValue, std::size_t subcell_offset>
struct _impl_evaluator<grid_t, count_neighbors<CellStateValue, von_neumann_4_neighbors>, subcell_offset> {
    
    template <int x_offset, int y_offset>
    using cell_at = _impl_evaluator<grid_t, neighbor_at<x_offset, y_offset>, subcell_offset>;

    CUDA_CALLABLE static int evaluate(state_t<grid_t> state) {
        auto target_state = _impl_evaluator<grid_t, CellStateValue, subcell_offset>::evaluate(state);

        auto top_c          = cell_at< 0, -1>::evaluate(state) == target_state ? 1 : 0;
        auto left_c         = cell_at<-1,  0>::evaluate(state) == target_state ? 1 : 0;
        auto right_c        = cell_at< 1,  0>::evaluate(state) == target_state ? 1 : 0;
        auto bottom_c       = cell_at< 0,  1>::evaluate(state) == target_state ? 1 : 0;

        return top_c + left_c + right_c + bottom_c;
    }
};

} // namespace cellib::evaluators::bit_array

#endif // CELLIB_BIT_ARRAY_EVALUATORS_HPP
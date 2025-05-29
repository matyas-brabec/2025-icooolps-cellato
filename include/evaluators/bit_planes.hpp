#ifndef CELLATO_EVALUATORS_BIT_PLANES_HPP
#define CELLATO_EVALUATORS_BIT_PLANES_HPP

#include <tuple>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <cstdint>
#include <algorithm>

#include "../core/ast.hpp"
#include "../core/vector_int.hpp"
#include "../memory/bit_planes_grid.hpp"
#include "../memory/grid_utils.hpp"
#include "../memory/interface.hpp"

// Use the same CUDA_CALLABLE definition as in standard evaluator
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace cellato::evaluators::bit_planes {

using namespace cellato::ast;
using namespace cellato::core::bitwise;
using namespace cellato::memory::grids::utils;

template <typename cell_row_type, typename state_dictionary_type, typename Expression>
struct evaluator {};

template <typename cell_row_type, typename state_dictionary_type>
using grid_cell_data_type = repeated_tuple_t<cell_row_type*, state_dictionary_type::needed_bits>;

template <typename cell_row_type, typename state_dictionary_type>
using state_t = cellato::memory::grids::point_in_grid<
    grid_cell_data_type<cell_row_type, state_dictionary_type>>;

template <typename cell_row_type, typename state_dictionary_type, auto Value>
struct evaluator<cell_row_type, state_dictionary_type, constant<Value>> {
    CUDA_CALLABLE static auto evaluate(state_t<cell_row_type, state_dictionary_type> /* state */) {
        return vector_int_factory::from_constant<cell_row_type, Value>();
    }
};

template <typename cell_row_type, typename state_dictionary_type, auto Value>
struct evaluator<cell_row_type, state_dictionary_type, state_constant<Value>> {
    CUDA_CALLABLE static auto evaluate(state_t<cell_row_type, state_dictionary_type> /* state */) {
        constexpr auto index = state_dictionary_type::state_to_index(Value);
        return vector_int_factory::from_constant<cell_row_type, index>();
    }
};

template <typename cell_row_type, typename state_dictionary_type, typename Condition, typename Then, typename Else>
struct evaluator<cell_row_type, state_dictionary_type, if_then_else<Condition, Then, Else>> {

    template <typename E>
    using evaluator_t = evaluator<cell_row_type, state_dictionary_type, E>;

    CUDA_CALLABLE static auto evaluate(state_t<cell_row_type, state_dictionary_type> state) {
        auto condition = evaluator_t<Condition>::evaluate(state);
        auto then_part = evaluator_t<Then>::evaluate(state);
        auto else_part = evaluator_t<Else>::evaluate(state);

        auto masked_then = then_part.mask_out_columns(condition);
        auto masked_else = else_part.mask_out_columns(~condition);

        return masked_then.get_ored(masked_else);
    }
};

template <typename cell_row_type, typename state_dictionary_type, typename Left, typename Right>
struct evaluator<cell_row_type, state_dictionary_type, and_<Left, Right>> {

    template <typename E>
    using evaluator_t = evaluator<cell_row_type, state_dictionary_type, E>;

    CUDA_CALLABLE static cell_row_type evaluate(state_t<cell_row_type, state_dictionary_type> state) {
        cell_row_type left = evaluator_t<Left>::evaluate(state);
        cell_row_type right = evaluator_t<Right>::evaluate(state);

        return left & right;
    }
};

template <typename cell_row_type, typename state_dictionary_type, typename Left, typename Right>
struct evaluator<cell_row_type, state_dictionary_type, or_<Left, Right>> {

    template <typename E>
    using evaluator_t = evaluator<cell_row_type, state_dictionary_type, E>;

    CUDA_CALLABLE static cell_row_type evaluate(state_t<cell_row_type, state_dictionary_type> state) {
        cell_row_type left = evaluator_t<Left>::evaluate(state);
        cell_row_type right = evaluator_t<Right>::evaluate(state);

        return left | right;
    }
};

template <typename cell_row_type, typename state_dictionary_type, typename Left, typename Right>
struct evaluator<cell_row_type, state_dictionary_type, equals<Left, Right>> {

    template <typename E>
    using evaluator_t = evaluator<cell_row_type, state_dictionary_type, E>;

    CUDA_CALLABLE static auto evaluate(state_t<cell_row_type, state_dictionary_type> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        auto right = evaluator_t<Right>::evaluate(state);

        return left.equals_to(right);
    }
};

template <typename cell_row_type, typename state_dictionary_type, typename Left, typename Right>
struct evaluator<cell_row_type, state_dictionary_type, greater_than<Left, Right>> {
    CUDA_CALLABLE static cell_row_type evaluate(state_t<cell_row_type, state_dictionary_type> state) {
        auto left = evaluator<cell_row_type, state_dictionary_type, Left>::evaluate(state);
        auto right = evaluator<cell_row_type, state_dictionary_type, Right>::evaluate(state);

        return left.greater_than(right);
    }
};

template <typename cell_row_type, typename state_dictionary_type, typename Left, typename Right>
struct evaluator<cell_row_type, state_dictionary_type, not_equals<Left, Right>> {

    template <typename E>
    using evaluator_t = evaluator<cell_row_type, state_dictionary_type, E>;

    CUDA_CALLABLE static bool evaluate(state_t<cell_row_type, state_dictionary_type> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        auto right = evaluator_t<Right>::evaluate(state);

        return left.not_equal_to(right);
    }
};

template <typename cell_row_type, typename state_dictionary_type, int x_offset, int y_offset>
struct evaluator<cell_row_type, state_dictionary_type, neighbor_at<x_offset, y_offset>> {
    static constexpr int vector_width_bits = sizeof(cell_row_type) * 8;

    using eval_state_t = state_t<cell_row_type, state_dictionary_type>;

    constexpr static std::size_t x_offset_unsigned = static_cast<std::size_t>(x_offset);
    constexpr static std::size_t y_offset_unsigned = static_cast<std::size_t>(y_offset);

    CUDA_CALLABLE static auto evaluate(eval_state_t state) {
        auto center = get_center_vector_int(state);

        if constexpr (x_offset == 0) {
            return center;
        }

        auto neighbor = get_neighbor_vector_int(state);

        auto shifted_center = shift_center(center);
        auto shifted_neighbor = shift_neighbor(neighbor);

        return shifted_center.get_ored(shifted_neighbor);
    }

  private:
    using vint = vector_int<cell_row_type, state_dictionary_type::needed_bits>;

    CUDA_CALLABLE static vint shift_center(vint center) {
        if constexpr (x_offset > 0) {
            return center.template get_right_shifted_vector<x_offset>();
        } else if constexpr (x_offset < 0) {
            return center.template get_left_shifted_vector<-x_offset>();
        } else {
            #ifndef __CUDA_ARCH__
            throw std::logic_error("Invalid x_offset value");
            #else
            // In CUDA device code, we can't throw exceptions
            // Just return the unshifted center as a fallback
            return center;
            #endif
        }
    }

    CUDA_CALLABLE static vint shift_neighbor(vint neighbor) {
        if constexpr (x_offset > 0) {
            return neighbor.template get_left_shifted_vector<vector_width_bits - x_offset>();
        } else if constexpr (x_offset < 0) {
            return neighbor.template get_right_shifted_vector<vector_width_bits + x_offset>();
        } else {
            #ifndef __CUDA_ARCH__
            throw std::logic_error("Invalid x_offset value");
            #else
            // In CUDA device code, we can't throw exceptions
            return neighbor;
            #endif
        }
    }

    CUDA_CALLABLE static vint get_center_vector_int(eval_state_t state) {
        auto x = state.position.x;
        auto y = state.position.y;
        auto idx = state.properties.idx(x, y + y_offset_unsigned);

        return vector_int_factory::load_from<cell_row_type>(state.grid, idx);
    }

    CUDA_CALLABLE static vint get_neighbor_vector_int(eval_state_t state) {
        auto x = state.position.x;
        auto y = state.position.y;
        auto idx = state.properties.idx(x + x_offset_unsigned, y + y_offset_unsigned);

        return vector_int_factory::load_from<cell_row_type>(state.grid, idx);
    }
};

template <typename cell_row_type, typename state_dictionary_type, typename cell_state_type, cell_state_type CellStateValue>
struct evaluator<
    cell_row_type, state_dictionary_type,
    count_neighbors<
        state_constant<CellStateValue>,
        moore_8_neighbors>> {

    template <typename E>
    using evaluator_t = evaluator<cell_row_type, state_dictionary_type, E>;

    CUDA_CALLABLE static vector_int<cell_row_type, 4> evaluate(state_t<cell_row_type, state_dictionary_type> state) {
        constexpr auto cell_state = state_dictionary_type::state_to_index(CellStateValue);

        auto top_left_c     = evaluator_t<neighbor_at<-1, -1>>::evaluate(state).template equals_to<cell_state>();
        auto top_c          = evaluator_t<neighbor_at< 0, -1>>::evaluate(state).template equals_to<cell_state>();
        auto top_right_c    = evaluator_t<neighbor_at< 1, -1>>::evaluate(state).template equals_to<cell_state>();
        auto left_c         = evaluator_t<neighbor_at<-1,  0>>::evaluate(state).template equals_to<cell_state>();
        auto right_c        = evaluator_t<neighbor_at< 1,  0>>::evaluate(state).template equals_to<cell_state>();
        auto bottom_left_c  = evaluator_t<neighbor_at<-1,  1>>::evaluate(state).template equals_to<cell_state>();
        auto bottom_c       = evaluator_t<neighbor_at< 0,  1>>::evaluate(state).template equals_to<cell_state>();
        auto bottom_right_c = evaluator_t<neighbor_at< 1,  1>>::evaluate(state).template equals_to<cell_state>();

        auto top_left       = vector_int_factory::from_condition_result<cell_row_type>(top_left_c);
        auto top            = vector_int_factory::from_condition_result<cell_row_type>(top_c);
        auto top_right      = vector_int_factory::from_condition_result<cell_row_type>(top_right_c);
        auto left           = vector_int_factory::from_condition_result<cell_row_type>(left_c);
        auto right          = vector_int_factory::from_condition_result<cell_row_type>(right_c);
        auto bottom_left    = vector_int_factory::from_condition_result<cell_row_type>(bottom_left_c);
        auto bottom         = vector_int_factory::from_condition_result<cell_row_type>(bottom_c);
        auto bottom_right   = vector_int_factory::from_condition_result<cell_row_type>(bottom_right_c);

        return top_left.template to_vector_with_bits<2>()
            .get_added(top)
            .get_added(top_right).template to_vector_with_bits<3>()
            .get_added(left)
            .get_added(right)
            .get_added(bottom_left)
            .get_added(bottom).template to_vector_with_bits<4>()
            .get_added(bottom_right);
    }
};

template <typename cell_row_type, typename state_dictionary_type, typename cell_state_type, cell_state_type CellStateValue>
struct evaluator<
    cell_row_type, state_dictionary_type,
    count_neighbors<
        state_constant<CellStateValue>,
        von_neumann_4_neighbors>> {

    template <typename E>
    using evaluator_t = evaluator<cell_row_type, state_dictionary_type, E>;

    CUDA_CALLABLE static vector_int<cell_row_type, 3> evaluate(state_t<cell_row_type, state_dictionary_type> state) {
        constexpr auto cell_state = state_dictionary_type::state_to_index(CellStateValue);

        // Get the four neighbors (top, right, bottom, left)
        auto top_c          = evaluator_t<neighbor_at< 0, -1>>::evaluate(state).template equals_to<cell_state>();
        auto right_c        = evaluator_t<neighbor_at< 1,  0>>::evaluate(state).template equals_to<cell_state>();
        auto bottom_c       = evaluator_t<neighbor_at< 0,  1>>::evaluate(state).template equals_to<cell_state>();
        auto left_c         = evaluator_t<neighbor_at<-1,  0>>::evaluate(state).template equals_to<cell_state>();

        // Convert condition results to vector_int
        auto top            = vector_int_factory::from_condition_result<cell_row_type>(top_c);
        auto right          = vector_int_factory::from_condition_result<cell_row_type>(right_c);
        auto bottom         = vector_int_factory::from_condition_result<cell_row_type>(bottom_c);
        auto left           = vector_int_factory::from_condition_result<cell_row_type>(left_c);

        // Add the counts using 3 bits (since max count is 4)
        return top.template to_vector_with_bits<3>()
            .get_added(right)
            .get_added(bottom)
            .get_added(left);
    }
};

} // namespace cellato::evaluators::bit_planes

#endif // CELLATO_EVALUATORS_BIT_PLANES_HPP
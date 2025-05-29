#ifndef CELLIB_STANDARD_EVALUATORS_HPP
#define CELLIB_STANDARD_EVALUATORS_HPP

#include <cstddef>

#include <utility>

#include "../core/ast.hpp"
#include "../memory/interface.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace cellib::evaluators::standard {

using namespace cellib::ast;
using namespace cellib::memory;

template <typename cell_type, typename Expression, typename cell_ptr_type = cell_type*>
struct evaluator {};

template <typename cell_type, typename cell_ptr_type = cell_type*>
using state_t = grids::point_in_grid<cell_ptr_type>;

template <typename cell_type, typename cell_ptr_type, auto Value>
struct evaluator<cell_type, constant<Value>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> /* state */) {
        return Value;
    }
};

template <typename cell_type, typename cell_ptr_type, auto Value>
struct evaluator<cell_type, state_constant<Value>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> /* state */) {
        return Value;
    }
};

template <typename cell_type, typename cell_ptr_type, typename Condition, typename Then, typename Else>
struct evaluator<cell_type, if_then_else<Condition, Then, Else>, cell_ptr_type> {
    static CUDA_CALLABLE cell_type evaluate(state_t<cell_type, cell_ptr_type> state) {
        if (evaluator<cell_type, Condition, cell_ptr_type>::evaluate(state)) {
            return evaluator<cell_type, Then, cell_ptr_type>::evaluate(state);
        } else {
            return evaluator<cell_type, Else, cell_ptr_type>::evaluate(state);
        }
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, and_<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE bool evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) && 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, or_<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE bool evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) || 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, equals<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE bool evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) == 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, greater_than<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE bool evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) > 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, not_equals<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE bool evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) != 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, int x_offset, int y_offset>
struct evaluator<cell_type, neighbor_at<x_offset, y_offset>, cell_ptr_type> {
    static CUDA_CALLABLE cell_type evaluate(state_t<cell_type, cell_ptr_type> state) {
        return state.grid[(state.position.x + x_offset) + (state.position.y + y_offset) * state.properties.x_size];
    }
};

template <typename cell_type, typename cell_ptr_type, typename CellStateValue>
struct evaluator<cell_type, count_neighbors<CellStateValue, moore_8_neighbors>, cell_ptr_type> {
    static CUDA_CALLABLE int evaluate(state_t<cell_type, cell_ptr_type> state) {
        auto target_value = evaluator<cell_type, CellStateValue, cell_ptr_type>::evaluate(state);

        return [target_value, state, x = state.position.x, y = state.position.y, x_size = state.properties.x_size]<std::size_t... I> (std::index_sequence<I...>) {
            constexpr int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
            constexpr int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

            return (... + (state.grid[(x + dx[I]) + (y + dy[I]) * x_size] == target_value));
        }(std::make_index_sequence<8>{});
    }
};

template <typename cell_type, typename cell_ptr_type, typename CellStateValue>
struct evaluator<cell_type, count_neighbors<CellStateValue, von_neumann_4_neighbors>, cell_ptr_type> {
    static CUDA_CALLABLE int evaluate(state_t<cell_type, cell_ptr_type> state) {
        auto target_value = evaluator<cell_type, CellStateValue, cell_ptr_type>::evaluate(state);

        return [target_value, state, x = state.position.x, y = state.position.y, x_size = state.properties.x_size]<std::size_t... I> (std::index_sequence<I...>) {
            constexpr int dx[] = {0, 0, 1, -1};
            constexpr int dy[] = {1, -1, 0, 0};

            return (... + (state.grid[(x + dx[I]) + (y + dy[I]) * x_size] == target_value));
        }(std::make_index_sequence<4>{});
    }
};

// example of nested specialization for count_neighbors with moore_8_neighbors for better performance

template <typename cell_type, typename cell_ptr_type, typename CellStateValue>
struct evaluator<cell_type, 
        greater_than< 
            count_neighbors<CellStateValue, moore_8_neighbors>, 
            state_constant<0>
        >, cell_ptr_type> {
    static CUDA_CALLABLE int evaluate(state_t<cell_type, cell_ptr_type> state) {
        auto target_value = evaluator<cell_type, CellStateValue, cell_ptr_type>::evaluate(state);

        return
            state.grid[(state.position.x - 1) + (state.position.y - 1) * state.properties.x_size] == target_value ||
            state.grid[(state.position.x - 1) + (state.position.y    ) * state.properties.x_size] == target_value ||
            state.grid[(state.position.x - 1) + (state.position.y + 1) * state.properties.x_size] == target_value ||
            state.grid[(state.position.x    ) + (state.position.y - 1) * state.properties.x_size] == target_value ||
            state.grid[(state.position.x    ) + (state.position.y + 1) * state.properties.x_size] == target_value ||
            state.grid[(state.position.x + 1) + (state.position.y - 1) * state.properties.x_size] == target_value ||
            state.grid[(state.position.x + 1) + (state.position.y    ) * state.properties.x_size] == target_value ||
            state.grid[(state.position.x + 1) + (state.position.y + 1) * state.properties.x_size] == target_value;
    }
};

} // namespace cellib::evaluators::standard

#endif // CELLIB_STANDARD_EVALUATORS_HPP
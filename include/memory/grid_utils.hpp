#ifndef GRID_UTILS_HPP
#define GRID_UTILS_HPP

#include <vector>
#include <tuple>
#include <random>  // For mt19937 and uniform_real_distribution
#include <iostream>
#include <cmath>   // For abs function

namespace cellib::memory::grids::utils {

// Generate a random grid with specified distribution of cell states
// Each tuple in probabilities contains (state, probability)
// Probabilities should sum to 1.0
template <typename CellState>
void generate_random_grid(
    std::vector<CellState>& grid,
    std::size_t height, std::size_t width,
    const std::vector<std::tuple<CellState, double>>& probabilities,
    unsigned int seed = 12345) {

    // Verify probabilities sum to approximately 1.0 (allowing for small floating point errors)
    double sum = 0.0;
    for (const auto& [state, prob] : probabilities) {
        sum += prob;
    }

    if (std::abs(sum - 1.0) > 0.001) {
        std::cerr << "Warning: Probabilities sum to " << sum << " instead of 1.0" << std::endl;
    }

    // Create cumulative distribution
    std::vector<std::tuple<CellState, double>> cumulative_dist;
    double cumulative = 0.0;

    for (const auto& [state, prob] : probabilities) {
        cumulative += prob;
        cumulative_dist.emplace_back(state, cumulative);
    }

    // Initialize random number generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Fill grid with random values based on the distribution
    for (std::size_t i = 0; i < height * width; ++i) {
        double r = dist(rng);

        // Find the first state with cumulative probability > r
        for (const auto& [state, threshold] : cumulative_dist) {
            if (r <= threshold) {
                grid[i] = state;
                break;
            }
        }
    }
}

// Shorthand for common case with just two states
template <typename CellState>
void generate_random_grid(
    std::vector<CellState>& grid,
    std::size_t height, std::size_t width,
    CellState primary_state, double primary_probability,
    CellState secondary_state,
    unsigned int seed = 12345) {

    std::vector<std::tuple<CellState, double>> probabilities = {
        {primary_state, primary_probability},
        {secondary_state, 1.0 - primary_probability}
    };

    generate_random_grid(grid, height, width, probabilities, seed);
}


template <typename T, std::size_t I>
using always_vector_t = std::vector<T>;

template <typename T, std::size_t... Is>
auto repeat_vector_type(std::index_sequence<Is...>) -> std::tuple<always_vector_t<T, Is>...>;

template <typename T, std::size_t N>
using repeated_vector_tuple = decltype(repeat_vector_type<T>(std::make_index_sequence<N>{}));

template <typename T, std::size_t I>
using always_type_t = T;

template <typename T, std::size_t... Is>
auto repeat_tuple_type(std::index_sequence<Is...>) -> std::tuple<always_type_t<T, Is>...>;

template <typename T, std::size_t N>
using repeated_tuple_t = decltype(repeat_tuple_type<T>(std::make_index_sequence<N>{}));


}

#endif // GRID_UTILS_HPP
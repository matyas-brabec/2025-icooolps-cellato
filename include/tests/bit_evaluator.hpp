#ifndef CELLATO_TESTS_BIT_EVALUATOR_HPP
#define CELLATO_TESTS_BIT_EVALUATOR_HPP

#include "manager.hpp"
#include "../evaluators/bit_planes.hpp"
#include "../core/ast.hpp"
#include "../core/vector_int.hpp"
#include <vector>
#include <tuple>

namespace cellato::tests {

// Define an enum for testing
enum class BitEvalTestState {
    DEAD,
    ALIVE,
    DYING
};

// Define the state dictionary for testing
using BitEvalTestDict = cellato::memory::grids::state_dictionary<
    BitEvalTestState::DEAD, 
    BitEvalTestState::ALIVE, 
    BitEvalTestState::DYING
>;

// Create state constants for tests
using dead = cellato::ast::state_constant<BitEvalTestState::DEAD>;
using alive = cellato::ast::state_constant<BitEvalTestState::ALIVE>;
using dying = cellato::ast::state_constant<BitEvalTestState::DYING>;

// Helper for tests to avoid long type names
template <typename Algorithm>
using bit_evaluator_t = cellato::evaluators::bit_planes::evaluator<uint8_t, BitEvalTestDict, Algorithm>;

class bit_evaluator_test_suite : public test_suite {
public:
    std::string name() const override {
        return "BitEvaluator";
    }

    test_result run() override {
        test_result result;
        test_case tc(result, true);

        // Run bit evaluator tests
        test_neighbor_accessor(tc);
        test_neighborhood_sum(tc);
        test_if_then_else(tc);
        
        return result;
    }

private:
    // Test neighbor accessors
    void test_neighbor_accessor(test_case& tc) {
        std::cout << BLUE << "\n--- Testing neighbor accessor functionality ---" << RESET << std::endl;
        
        using namespace cellato::ast;
        
        // Define neighbor accessors
        using top_left = neighbor_at<-1, -1>;
        using top = neighbor_at<0, -1>;
        using top_right = neighbor_at<1, -1>;
        // using left = neighbor_at<-1, 0>;
        using self = neighbor_at<0, 0>;
        using right = neighbor_at<1, 0>;
        // using bottom_left = neighbor_at<-1, 1>;
        using bottom = neighbor_at<0, 1>;
        // using bottom_right = neighbor_at<1, 1>;

        // Create test grid data - setting bits to represent different states
        std::vector<uint8_t> grid_0th_bit = {
            0b00000000, 0b00000000, 0b00000000,
            0b10000000, 0b00000010, 0b00000000,
            0b10000000, 0b00000101, 0b00000000, 
            0b10000000, 0b00001000, 0b00000000,
        };

        std::vector<uint8_t> grid_1st_bit = {
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b00000000, 0b00000000,
        };

        // Create grid configuration
        std::tuple<uint8_t*, uint8_t*> grid = { grid_0th_bit.data(), grid_1st_bit.data() };
        
        // Set up the state for evaluation
        cellato::memory::grids::point_in_grid<decltype(grid)> state;
        state.grid = grid;
        state.properties.x_size = 3;
        state.properties.y_size = 4;
        state.position.x = 1;
        state.position.y = 2;

        // Test various neighbor accessors
        auto top_left_res = bit_evaluator_t<top_left>::evaluate(state);
        tc.assert_true(top_left_res.get_at(0) == 1, "Top-left should access the correct cell");
        
        auto top_res = bit_evaluator_t<top>::evaluate(state);
        tc.assert_true(top_res.get_at(1) == 1, "Top should access the correct cell");
        
        auto self_res = bit_evaluator_t<self>::evaluate(state);
        tc.assert_true(self_res.get_at(0) == 1 && self_res.get_at(2) == 1, 
                      "Self accessor should return the current cell state");
                      
        auto bottom_res = bit_evaluator_t<bottom>::evaluate(state);
        tc.assert_true(bottom_res.get_at(3) == 1, "Bottom should access the correct cell");

        // Test that other neighbors are correctly empty
        auto top_right_res = bit_evaluator_t<top_right>::evaluate(state);
        tc.assert_true(top_right_res.get_at(0) == 1, "Top-right should access the correct cell");
        
        auto right_res = bit_evaluator_t<right>::evaluate(state);
        tc.assert_true(right_res.get_at(0) == 0, "Right should be empty");
    }

    // Test neighborhood sum functionality
    void test_neighborhood_sum(test_case& tc) {
        std::cout << BLUE << "\n--- Testing neighborhood sum functionality ---" << RESET << std::endl;
        
        using namespace cellato::ast;
        
        // Define a sum expression
        using sum = count_neighbors<alive, moore_8_neighbors>;

        // Create test grid data with some alive cells around position (1,2)
        std::vector<uint8_t> grid_0th_bit = {
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b00000011, 0b00000000,
            0b10000000, 0b00000010, 0b00000000,
            0b00000000, 0b00000011, 0b00000000,
        };

        std::vector<uint8_t> grid_1st_bit = {
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b00000001, 0b00000000,
        };

        // Create grid configuration
        std::tuple<uint8_t*, uint8_t*> grid = { grid_0th_bit.data(), grid_1st_bit.data() };
        
        // Set up the state for evaluation
        cellato::memory::grids::point_in_grid<decltype(grid)> state;
        state.grid = grid;
        state.properties.x_size = 3;
        state.properties.y_size = 4;
        state.position.x = 1;
        state.position.y = 2;

        // Evaluate the neighborhood sum
        auto result = bit_evaluator_t<sum>::evaluate(state);
        
        // We should have some alive cells in the neighborhood
        // The specific count depends on the bit patterns set above
        // Based on the binary patterns, we expect several alive cells
        tc.assert_true(result.get_at(0) > 0, "Should find alive cells in the neighborhood");
        
        // Check the type of the result
        tc.assert_true(std::is_same_v<decltype(result), 
                                      cellato::core::bitwise::vector_int<uint8_t, 4>>, 
                      "Result should be a 4-bit vector_int for Moore-8 neighbors");
    }

    // Test if-then-else conditional functionality
    void test_if_then_else(test_case& tc) {
        std::cout << BLUE << "\n--- Testing if-then-else functionality ---" << RESET << std::endl;
        
        using namespace cellato::ast;
        
        // Define expressions for testing
        using current = current_state;
        using cell_is_dying = p<current, equals, dying>;
        
        // Define two equivalent if-then-else expressions
        using next_state = if_<cell_is_dying>::then_<dead>::else_<alive>;
        using next_state_no_sugar = if_then_else<cell_is_dying, dead, alive>;

        // Create test grid data with various cell states
        std::vector<uint8_t> grid_0th_bit = {
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b00000011, 0b00000000,
            0b10000000, 0b01010010, 0b00000000,
            0b00000000, 0b00000011, 0b00000000,
        };

        std::vector<uint8_t> grid_1st_bit = {
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b00000000, 0b00000000,
            0b00000000, 0b01100010, 0b00000000,
            0b00000000, 0b00000001, 0b00000000,
        };

        // Create grid configuration
        std::tuple<uint8_t*, uint8_t*> grid = { grid_0th_bit.data(), grid_1st_bit.data() };
        
        // Set up the state for evaluation
        cellato::memory::grids::point_in_grid<decltype(grid)> state;
        state.grid = grid;
        state.properties.x_size = 3;
        state.properties.y_size = 4;
        state.position.x = 1;
        state.position.y = 2;

        // Get current state at the test position
        // auto curr_state = bit_evaluator_t<current>::evaluate(state);
        
        // Check if any cells are in DYING state
        auto is_dying_result = bit_evaluator_t<cell_is_dying>::evaluate(state);
        
        // Evaluate the if-then-else expression
        auto result = bit_evaluator_t<next_state>::evaluate(state);
        auto result_no_sugar = bit_evaluator_t<next_state_no_sugar>::evaluate(state);
        
        // The results from both expressions should be identical
        tc.assert_true(result.equals_to(result_no_sugar) == 0xFF, 
                      "Sugar and non-sugar syntax should produce identical results");
        
        // Check the behavior of the if-then-else
        // Cells that were dying should now be dead, others should be alive
        for (int i = 0; i < 8; i++) {
            bool is_dying_at_i = ((is_dying_result >> i) & 1) != 0;
            bool is_dead_at_i = result.get_at(i) == BitEvalTestDict::state_to_index(BitEvalTestState::DEAD);
            bool is_alive_at_i = result.get_at(i) == BitEvalTestDict::state_to_index(BitEvalTestState::ALIVE);
            
            if (is_dying_at_i) {
                tc.assert_true(is_dead_at_i, "Cells that were dying should now be dead");
            } else {
                tc.assert_true(is_alive_at_i, "Cells that were not dying should now be alive");
            }
        }
    }
};

// Helper function to register the suite with the manager
inline void register_bit_evaluator_tests() {
    static bit_evaluator_test_suite suite;
    test_manager::instance().register_suite(&suite);
}

} // namespace cellato::tests

#endif // CELLATO_TESTS_BIT_EVALUATOR_HPP

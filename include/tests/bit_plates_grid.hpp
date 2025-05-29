#ifndef CELLIB_TESTS_BIT_PLATES_GRID_HPP
#define CELLIB_TESTS_BIT_PLATES_GRID_HPP

#include "manager.hpp"
#include "../memory/bit_plates_grid.hpp"
#include "../memory/grid_utils.hpp"
#include "../memory/state_dictionary.hpp"
#include <random>
#include <ctime>

namespace cellib::tests {
// Create a nested namespace for bit_plates tests to avoid conflicts
namespace bit_plates {

// Define an enum for testing
enum class TestCellState {
    DEAD,
    ALIVE,
    DYING
};

// Stream operator for TestCellState to help with test output
inline std::string to_string(const TestCellState& state) {
    switch (state) {
        case TestCellState::DEAD: return "DEAD";
        case TestCellState::ALIVE: return "ALIVE";
        case TestCellState::DYING: return "DYING";
        default: return "UNKNOWN";
    }
}

// Define the state dictionary for testing
using TestStateDictionary = cellib::memory::grids::state_dictionary<
    TestCellState::DEAD, 
    TestCellState::ALIVE, 
    TestCellState::DYING
>;

} // namespace bit_plates

class bit_plates_grid_test_suite : public test_suite {
public:
    std::string name() const override {
        return "BitPlatesGrid";
    }

    test_result run() override {
        test_result result;
        test_case tc(result, true);

        // Run all bit_plates_grid tests
        test_state_dictionary_basics(tc);
        test_state_dictionary_conversion(tc);
        test_bit_grid_sizes(tc);
        test_bit_grid_pattern(tc);
        test_bit_grid_complex(tc);
        test_get_cell(tc);
        test_to_original_representation_with_get_cell(tc);
        test_bit_grid_small_random(tc);  // Smaller scale for unit tests
        
        return result;
    }

private:
    // Test state_dictionary basic properties
    void test_state_dictionary_basics(test_case& tc) {
        std::cout << BLUE << "\n--- Testing state_dictionary basics ---" << RESET << std::endl;
        
        tc.assert_equal(3, bit_plates::TestStateDictionary::number_of_values, "Dictionary should have 3 values");
        tc.assert_equal(2, bit_plates::TestStateDictionary::needed_bits, "Should need 2 bits to represent 3 states");
    }

    // Test state_dictionary conversion functions
    void test_state_dictionary_conversion(test_case& tc) {
        std::cout << BLUE << "\n--- Testing state_dictionary conversion ---" << RESET << std::endl;
        
        tc.assert_equal(0, bit_plates::TestStateDictionary::state_to_index(bit_plates::TestCellState::DEAD), "DEAD should map to index 0");
        tc.assert_equal(1, bit_plates::TestStateDictionary::state_to_index(bit_plates::TestCellState::ALIVE), "ALIVE should map to index 1");
        tc.assert_equal(2, bit_plates::TestStateDictionary::state_to_index(bit_plates::TestCellState::DYING), "DYING should map to index 2");
        
        tc.assert_true(bit_plates::TestCellState::DEAD == bit_plates::TestStateDictionary::index_to_state(0), "Index 0 should map to DEAD");
        tc.assert_true(bit_plates::TestCellState::ALIVE == bit_plates::TestStateDictionary::index_to_state(1), "Index 1 should map to ALIVE");
        tc.assert_true(bit_plates::TestCellState::DYING == bit_plates::TestStateDictionary::index_to_state(2), "Index 2 should map to DYING");
        
        bool exception_thrown = false;
        try {
            bit_plates::TestStateDictionary::state_to_index(static_cast<bit_plates::TestCellState>(99));
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for invalid state");
        
        exception_thrown = false;
        try {
            bit_plates::TestStateDictionary::index_to_state(99);
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for invalid index");
    }

    // Test bit_plates_grid construction and size methods
    void test_bit_grid_sizes(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_plates_grid sizes ---" << RESET << std::endl;
        
        // Create a 2x3 grid (2 rows, 3 word columns)
        const size_t height = 2;
        const size_t width_words = 3;
        const size_t word_bits = sizeof(uint8_t) * 8;
        const size_t width = width_words * word_bits;
        
        // Initialize grid with DEAD cells
        std::vector<bit_plates::TestCellState> input_grid(height * width, bit_plates::TestCellState::DEAD);
        
        cellib::memory::grids::bit_plates::grid<uint8_t, bit_plates::TestStateDictionary> grid(height, width, input_grid.data());

        tc.assert_equal(width, grid.x_size_original(), "Grid width should match original width");
        tc.assert_equal(height, grid.y_size_original(), "Grid height should match original height");
        tc.assert_equal(width_words, grid.x_size_physical(), "Physical width should be original width / bits_per_word");
        tc.assert_equal(height, grid.y_size_physical(), "Physical height should match original height");
    }

    // Test bit_plates_grid storage and retrieval of patterns
    void test_bit_grid_pattern(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_plates_grid pattern storage and retrieval ---" << RESET << std::endl;
        
        // Create a 1x1 grid (1 row, 1 word)
        // For uint8_t, this stores 8 cells in a row
        std::vector<bit_plates::TestCellState> input_grid = {
            bit_plates::TestCellState::DEAD, bit_plates::TestCellState::ALIVE, bit_plates::TestCellState::DYING, bit_plates::TestCellState::DEAD,
            bit_plates::TestCellState::ALIVE, bit_plates::TestCellState::DEAD, bit_plates::TestCellState::ALIVE, bit_plates::TestCellState::DYING
        };

        cellib::memory::grids::bit_plates::grid<uint8_t, bit_plates::TestStateDictionary> grid(1, 8, input_grid.data());
        
        // Reconstruct and verify
        auto result = grid.to_original_representation();
        
        tc.assert_equal(size_t{8}, result.size(), "Result should have 8 cells");
        
        // Check each cell matches what we put in
        for (size_t i = 0; i < 8; ++i) {
            tc.assert_true(input_grid[i] == result[i], "Cell " + std::to_string(i) + " should match input");
        }
    }

    // Test bit_plates_grid with larger grid and complex pattern
    void test_bit_grid_complex(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_plates_grid with complex pattern ---" << RESET << std::endl;
        
        // Create a 2x2 grid (2 rows, 2 word columns) for 2x16 cells with uint8_t
        const size_t height = 2;
        const size_t width_words = 2;
        const size_t word_bits = sizeof(uint8_t) * 8;
        const size_t width = width_words * word_bits;
        std::vector<bit_plates::TestCellState> input_grid(height * width, bit_plates::TestCellState::DEAD);
        
        // Set specific cells to create a pattern
        // Row 0, positions 0, 3, 7 are ALIVE
        // Row 1, positions 1, 4, 9 are DYING
        input_grid[0] = bit_plates::TestCellState::ALIVE;
        input_grid[3] = bit_plates::TestCellState::ALIVE;
        input_grid[7] = bit_plates::TestCellState::ALIVE;
        input_grid[width + 1] = bit_plates::TestCellState::DYING;
        input_grid[width + 4] = bit_plates::TestCellState::DYING;
        input_grid[width + 9] = bit_plates::TestCellState::DYING;
        
        cellib::memory::grids::bit_plates::grid<uint8_t, bit_plates::TestStateDictionary> grid(height, width, input_grid.data());
        
        // Verify the reconstruction
        auto result = grid.to_original_representation();
        
        tc.assert_true(bit_plates::TestCellState::ALIVE == result[0], "Cell (0,0) should be ALIVE");
        tc.assert_true(bit_plates::TestCellState::ALIVE == result[3], "Cell (0,3) should be ALIVE");
        tc.assert_true(bit_plates::TestCellState::ALIVE == result[7], "Cell (0,7) should be ALIVE");
        tc.assert_true(bit_plates::TestCellState::DYING == result[width + 1], "Cell (1,1) should be DYING");
        tc.assert_true(bit_plates::TestCellState::DYING == result[width + 4], "Cell (1,4) should be DYING");
        tc.assert_true(bit_plates::TestCellState::DYING == result[width + 9], "Cell (1,9) should be DYING");
    }

    // Test get_cell function
    void test_get_cell(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_plates_grid get_cell function ---" << RESET << std::endl;
        
        // Create a 2x2 grid (2 rows, 2 word columns) with a specific pattern
        const size_t height = 2;
        const size_t width_words = 2;
        const size_t word_bits = sizeof(uint8_t) * 8;
        const size_t width = width_words * word_bits;
        std::vector<bit_plates::TestCellState> input_grid(height * width, bit_plates::TestCellState::DEAD);
        
        // Set specific cells based on their (x, y) coordinates
        // Row 0
        input_grid[0] = bit_plates::TestCellState::ALIVE;                 // (0,0)
        input_grid[3] = bit_plates::TestCellState::DYING;                 // (3,0)
        
        // Row 1 - offset by width
        input_grid[width + 1] = bit_plates::TestCellState::ALIVE;         // (1,1)
        input_grid[width + 7] = bit_plates::TestCellState::DYING;         // (7,1)
        input_grid[width + 9] = bit_plates::TestCellState::ALIVE;         // (9,1)
        
        cellib::memory::grids::bit_plates::grid<uint8_t, bit_plates::TestStateDictionary> grid(height, width, input_grid.data());
        
        // Test specific cell retrievals
        tc.assert_true(bit_plates::TestCellState::ALIVE == grid.get_cell(0, 0), "Cell (0,0) should be ALIVE");
        tc.assert_true(bit_plates::TestCellState::DEAD == grid.get_cell(1, 0), "Cell (1,0) should be DEAD");
        tc.assert_true(bit_plates::TestCellState::DYING == grid.get_cell(3, 0), "Cell (3,0) should be DYING");
        tc.assert_true(bit_plates::TestCellState::ALIVE == grid.get_cell(1, 1), "Cell (1,1) should be ALIVE");
        tc.assert_true(bit_plates::TestCellState::DYING == grid.get_cell(7, 1), "Cell (7,1) should be DYING");
        tc.assert_true(bit_plates::TestCellState::ALIVE == grid.get_cell(9, 1), "Cell (9,1) should be ALIVE");
        
        // Test bounds checking
        bool exception_thrown = false;
        try {
            grid.get_cell(width, 0);
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for out of bounds x coordinate");
        
        exception_thrown = false;
        try {
            grid.get_cell(0, height);
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for out of bounds y coordinate");
    }

    // Test that to_original_representation uses get_cell correctly
    void test_to_original_representation_with_get_cell(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_plates_grid to_original_representation with get_cell ---" << RESET << std::endl;
        
        // Create a small grid
        const size_t height = 4;
        const size_t width_words = 2;
        const size_t word_bits = sizeof(uint8_t) * 8;
        const size_t width = width_words * word_bits;
        const size_t total_cells = height * width;
        
        // Generate patterned cell states
        std::vector<bit_plates::TestCellState> input_grid(total_cells);
        for (size_t i = 0; i < input_grid.size(); ++i) {
            switch (i % 3) {
                case 0: input_grid[i] = bit_plates::TestCellState::DEAD; break;
                case 1: input_grid[i] = bit_plates::TestCellState::ALIVE; break;
                case 2: input_grid[i] = bit_plates::TestCellState::DYING; break;
            }
        }
        
        cellib::memory::grids::bit_plates::grid<uint8_t, bit_plates::TestStateDictionary> grid(height, width, input_grid.data());
        
        // Compare direct get_cell with to_original_representation results
        auto result = grid.to_original_representation();
        
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = y * width + x;
                bit_plates::TestCellState from_get_cell = grid.get_cell(x, y);
                bit_plates::TestCellState from_representation = result[idx];
                
                tc.assert_true(from_get_cell == from_representation, 
                    "get_cell and to_original_representation should return the same value at (" + 
                    std::to_string(x) + "," + std::to_string(y) + ")");
            }
        }
    }

    // Test bit_plates_grid with a small random grid (for unit tests)
    void test_bit_grid_small_random(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_plates_grid with small random pattern ---" << RESET << std::endl;
        
        // Create a smaller random grid for unit tests
        const size_t height = 10;
        const size_t width_words = 4;
        const size_t word_bits = sizeof(uint8_t) * 8;
        const size_t width = width_words * word_bits;
        const size_t total_cells = height * width;
        
        std::cout << "  Generating random grid with " << total_cells << " cells..." << std::endl;
        std::vector<bit_plates::TestCellState> input_grid(total_cells);
        
        // Initialize with random values
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dist(0, 2);
        
        for (size_t i = 0; i < input_grid.size(); ++i) {
            int random_value = dist(rng);
            switch (random_value) {
                case 0: input_grid[i] = bit_plates::TestCellState::DEAD; break;
                case 1: input_grid[i] = bit_plates::TestCellState::ALIVE; break;
                case 2: input_grid[i] = bit_plates::TestCellState::DYING; break;
            }
        }
        
        std::cout << "  Creating bit plates grid..." << std::endl;
        cellib::memory::grids::bit_plates::grid<uint8_t, bit_plates::TestStateDictionary> grid(height, width, input_grid.data());
        
        std::cout << "  Converting back to original representation..." << std::endl;
        auto result = grid.to_original_representation();
        
        std::cout << "  Verifying results..." << std::endl;
        tc.assert_equal(input_grid.size(), result.size(), "Result size should match input size");
        
        // Check a subset of cells to avoid too many assertions
        const int check_interval = 5;
        
        for (size_t i = 0; i < input_grid.size(); i += check_interval) {
            tc.assert_true(input_grid[i] == result[i], 
                "Cell at index " + std::to_string(i) + " should match original");
        }
        
        // Also check the last cell
        tc.assert_true(input_grid[total_cells-1] == result[total_cells-1], 
            "Last cell should match original");
    }
};

// Helper function to register the suite with the manager
inline void register_bit_plates_grid_tests() {
    static bit_plates_grid_test_suite suite;
    test_manager::instance().register_suite(&suite);
}

} // namespace cellib::tests

#endif // CELLIB_TESTS_BIT_PLATES_GRID_HPP

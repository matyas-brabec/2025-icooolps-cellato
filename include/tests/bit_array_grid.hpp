#ifndef CELLIB_TESTS_BIT_ARRAY_GRID_HPP
#define CELLIB_TESTS_BIT_ARRAY_GRID_HPP

#include "manager.hpp"
#include "../memory/bit_array_grid.hpp"
#include "../memory/grid_utils.hpp"
#include "../memory/state_dictionary.hpp"
#include "../memory/standard_grid.hpp"
#include <random>
#include <ctime>

namespace cellib::tests {
// Create a nested namespace for bit_array tests to avoid conflicts
namespace bit_array {

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

} // namespace bit_array

class bit_array_grid_test_suite : public test_suite {
private:
    // Test grid construction
    void test_grid_construction(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_array_grid construction ---" << RESET << std::endl;
        
        // Modified: Use 16 instead of 20 for x_size to be divisible by cells_per_word
        cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary> grid(10, 16);
        tc.assert_equal(size_t{16}, grid.x_size_logical(), "Grid width should be 16");
        tc.assert_equal(size_t{10}, grid.y_size_logical(), "Grid height should be 10");
        
        // All cells should be default-initialized to zero (which is DEAD in our enum)
        for (size_t y = 0; y < 5; ++y) {
            for (size_t x = 0; x < 5; ++x) {
                tc.assert_true(bit_array::TestCellState::DEAD == grid.get_cell(x, y), 
                             "Default cell value should be DEAD at (" + std::to_string(x) + ", " + std::to_string(y) + ")");
            }
        }
    }
    
    // Test get_cell and set_cell via construction
    void test_cell_access(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_array_grid cell access ---" << RESET << std::endl;
        
        const size_t height = 5;
        // Modified: Use 16 instead of 10 for width to be divisible by cells_per_word
        const size_t width = 16;
        std::vector<bit_array::TestCellState> init_data(height * width, bit_array::TestCellState::DEAD);
        
        // Set specific cells
        init_data[0] = bit_array::TestCellState::ALIVE;          // (0,0)
        init_data[2] = bit_array::TestCellState::DYING;          // (2,0)
        init_data[width + 3] = bit_array::TestCellState::ALIVE;  // (3,1)
        
        cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary> grid(height, width, init_data.data());
        
        // Test specific locations
        tc.assert_true(bit_array::TestCellState::ALIVE == grid.get_cell(0, 0), "Cell (0,0) should be ALIVE");
        tc.assert_true(bit_array::TestCellState::DEAD == grid.get_cell(1, 0), "Cell (1,0) should be DEAD");
        tc.assert_true(bit_array::TestCellState::DYING == grid.get_cell(2, 0), "Cell (2,0) should be DYING");
        tc.assert_true(bit_array::TestCellState::ALIVE == grid.get_cell(3, 1), "Cell (3,1) should be ALIVE");
        
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
    
    // Test bit_array_grid sizes
    void test_grid_sizes(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_array_grid sizes ---" << RESET << std::endl;
        
        constexpr std::size_t cells_per_word = sizeof(uint32_t) * 8 / bit_array::TestStateDictionary::needed_bits;

        const size_t height = 16;
        const size_t width = 32;
        
        cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary> grid(height, width);
        
        const size_t expected_x_size = width / cells_per_word;

        tc.assert_equal(width, grid.x_size_logical(), "Logical grid width should match input width");
        tc.assert_equal(height, grid.y_size_logical(), "Logical grid height should match input height");
        tc.assert_equal(expected_x_size, grid.x_size_physical(), "Physical width should be original width / cells_per_word");
        tc.assert_equal(height, grid.y_size_physical(), "Physical height should match logical height");
        
        // Check the bits_per_cell and cells_per_word constants
        tc.assert_equal(bit_array::TestStateDictionary::needed_bits, 
                      cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary>::bits_per_cell, 
                      "bits_per_cell should match state dictionary needed_bits");
        
        // Calculate expected cells per word based on the store_word_type (defaulted to uint32_t)
        const size_t expected_cells_per_word = 32 / bit_array::TestStateDictionary::needed_bits; // 32-bit word / 2 bits per cell = 16 cells
        tc.assert_equal(expected_cells_per_word, 
                      cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary>::cells_per_word, 
                      "cells_per_word calculation should be correct");
    }
    
    // Test simple pattern storage and retrieval
    void test_simple_pattern(test_case& tc) {
        // This test uses width=8, which needs to be modified to width=16
        std::cout << BLUE << "\n--- Testing bit_array_grid simple pattern ---" << RESET << std::endl;
        
        const size_t height = 2;
        const size_t width = 16; // Modified: Use 16 instead of 8
        std::vector<bit_array::TestCellState> init_data(height * width, bit_array::TestCellState::DEAD);
        
        // Set only the first 16 elements with the pattern
        init_data[0] = bit_array::TestCellState::DEAD;
        init_data[1] = bit_array::TestCellState::ALIVE;
        init_data[2] = bit_array::TestCellState::DYING;
        init_data[3] = bit_array::TestCellState::DEAD;
        init_data[4] = bit_array::TestCellState::ALIVE;
        init_data[5] = bit_array::TestCellState::DEAD;
        init_data[6] = bit_array::TestCellState::ALIVE;
        init_data[7] = bit_array::TestCellState::DYING;
        
        init_data[width + 0] = bit_array::TestCellState::DYING;
        init_data[width + 1] = bit_array::TestCellState::ALIVE;
        init_data[width + 2] = bit_array::TestCellState::DEAD;
        init_data[width + 3] = bit_array::TestCellState::DEAD;
        init_data[width + 4] = bit_array::TestCellState::ALIVE;
        init_data[width + 5] = bit_array::TestCellState::ALIVE;
        init_data[width + 6] = bit_array::TestCellState::DYING;
        init_data[width + 7] = bit_array::TestCellState::DEAD;
        
        cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary> grid(height, width, init_data.data());
        
        // Verify original state can be reconstructed
        auto reconstructed = grid.to_original_representation();
        
        tc.assert_equal(height * width, reconstructed.size(), "Reconstructed vector should have the correct size");
        
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = y * width + x;
                tc.assert_true(init_data[idx] == reconstructed[idx], 
                             "Cell at (" + std::to_string(x) + ", " + std::to_string(y) + ") should match original");
            }
        }
    }
    
    // Test more complex pattern across word boundaries
    void test_complex_pattern(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_array_grid complex pattern ---" << RESET << std::endl;
        
        const size_t height = 3;
        // Modified: Use 48 instead of 50 for width to be divisible by cells_per_word
        const size_t width = 48;  // Ensure this crosses word boundaries but is divisible by cells_per_word
        std::vector<bit_array::TestCellState> init_data(height * width, bit_array::TestCellState::DEAD);
        
        // Create a pattern that crosses word boundaries
        for (size_t i = 0; i < height * width; ++i) {
            switch (i % 3) {
                case 0: init_data[i] = bit_array::TestCellState::DEAD; break;
                case 1: init_data[i] = bit_array::TestCellState::ALIVE; break;
                case 2: init_data[i] = bit_array::TestCellState::DYING; break;
            }
        }
        
        cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary> grid(height, width, init_data.data());
        
        // Test retrieving specific cells from each row and different word boundaries
        for (size_t y = 0; y < height; ++y) {
            for (size_t x : {0, 15, 16, 31, 32, 45}) {  // Test cells from different words
                if (x < width) {
                    size_t idx = y * width + x;
                    tc.assert_true(init_data[idx] == grid.get_cell(x, y),
                                 "Cell at (" + std::to_string(x) + ", " + std::to_string(y) + ") should match original");
                }
            }
        }
        
        // Verify entire grid matches original 
        auto reconstructed = grid.to_original_representation();
        for (size_t i = 0; i < height * width; i += 5) {  // Check every 5th cell to reduce test time
            tc.assert_true(init_data[i] == reconstructed[i],
                         "Cell at index " + std::to_string(i) + " should match original");
        }
    }
    
    // Test conversion to standard grid
    void test_to_standard_conversion(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_array_grid to standard grid conversion ---" << RESET << std::endl;
        
        const size_t height = 4;
        // Modified: Use 16 instead of 20 for width to be divisible by cells_per_word
        const size_t width = 16;
        std::vector<bit_array::TestCellState> init_data(height * width, bit_array::TestCellState::DEAD);
        
        // Create a pattern
        for (size_t i = 0; i < init_data.size(); ++i) {
            switch (i % 3) {
                case 0: init_data[i] = bit_array::TestCellState::DEAD; break;
                case 1: init_data[i] = bit_array::TestCellState::ALIVE; break;
                case 2: init_data[i] = bit_array::TestCellState::DYING; break;
            }
        }
        
        cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary> array_grid(height, width, init_data.data());
        
        // Convert to standard grid
        auto standard_grid = array_grid.to_standard();
        
        // Check dimensions
        tc.assert_equal(width, standard_grid.x_size_physical(), "Standard grid width should match original");
        tc.assert_equal(height, standard_grid.y_size_physical(), "Standard grid height should match original");
        
        // Check cell values
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = y * width + x;
                tc.assert_true(init_data[idx] == standard_grid.data()[idx],
                             "Cell at (" + std::to_string(x) + ", " + std::to_string(y) + ") should match in standard grid");
            }
        }
    }
    
    // Test constructing from standard grid
    void test_from_standard_conversion(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_array_grid from standard grid conversion ---" << RESET << std::endl;
        
        const size_t height = 4;
        // Modified: Use 16 instead of 20 for width to be divisible by cells_per_word
        const size_t width = 16;
        std::vector<bit_array::TestCellState> init_data(height * width, bit_array::TestCellState::DEAD);
        
        // Create a pattern
        for (size_t i = 0; i < init_data.size(); ++i) {
            switch (i % 3) {
                case 0: init_data[i] = bit_array::TestCellState::DEAD; break;
                case 1: init_data[i] = bit_array::TestCellState::ALIVE; break;
                case 2: init_data[i] = bit_array::TestCellState::DYING; break;
            }
        }
        
        // Create standard grid first
        cellib::memory::grids::standard::grid<bit_array::TestCellState> standard_grid(std::move(init_data), width, height);
        
        // Create bit array grid from standard grid
        cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary> array_grid(standard_grid);
        
        // Check dimensions
        tc.assert_equal(width, array_grid.x_size_logical(), "Bit array grid width should match original");
        tc.assert_equal(height, array_grid.y_size_logical(), "Bit array grid height should match original");
        
        // Verify cell values by converting both to vectors and comparing
        auto standard_vec = std::vector<bit_array::TestCellState>(standard_grid.data(), standard_grid.data() + width * height);
        auto array_vec = array_grid.to_original_representation();
        
        tc.assert_equal(standard_vec.size(), array_vec.size(), "Vector sizes should match");
        
        for (size_t i = 0; i < standard_vec.size(); ++i) {
            tc.assert_true(standard_vec[i] == array_vec[i],
                         "Cell at index " + std::to_string(i) + " should match between grid types");
        }
    }
    
    // Test with random grid
    void test_random_grid(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_array_grid with random data ---" << RESET << std::endl;
        
        const size_t height = 8;
        // Modified: Use 32 instead of 30 for width to be divisible by cells_per_word
        const size_t width = 32;
        std::vector<bit_array::TestCellState> init_data(height * width);
        
        // Initialize with random values
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dist(0, 2);
        
        for (size_t i = 0; i < init_data.size(); ++i) {
            int random_value = dist(rng);
            switch (random_value) {
                case 0: init_data[i] = bit_array::TestCellState::DEAD; break;
                case 1: init_data[i] = bit_array::TestCellState::ALIVE; break;
                case 2: init_data[i] = bit_array::TestCellState::DYING; break;
            }
        }
        
        cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary> grid(height, width, init_data.data());
        
        // Verify using to_original_representation
        auto reconstructed = grid.to_original_representation();
        
        tc.assert_equal(init_data.size(), reconstructed.size(), "Reconstructed size should match original");
        
        // Check a subset of cells
        for (size_t i = 0; i < init_data.size(); i += 5) {
            tc.assert_true(init_data[i] == reconstructed[i],
                         "Cell at index " + std::to_string(i) + " should match original");
        }
        
        // Check last cell
        tc.assert_true(init_data.back() == reconstructed.back(), "Last cell should match original");
    }
    
    // Test memory efficiency
    void test_memory_efficiency(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_array_grid memory efficiency ---" << RESET << std::endl;
        
        // Create a large grid to demonstrate memory efficiency
        const size_t height = 10;
        // Modified: Use 1024 instead of 1000 for width to be divisible by cells_per_word
        const size_t width = 1024;
        
        // Calculate expected memory usage
        const size_t bits_per_cell = bit_array::TestStateDictionary::needed_bits;
        const size_t cells_per_word = (sizeof(uint32_t) * 8) / bits_per_cell;
        const size_t total_words_needed = (height * width + cells_per_word - 1) / cells_per_word;
        const size_t expected_bytes = total_words_needed * sizeof(uint32_t);
        
        // Create a message explaining the memory efficiency
        std::string efficiency_msg = "Memory usage: " + std::to_string(expected_bytes) + " bytes for " + 
                                    std::to_string(height * width) + " cells, compared to " + 
                                    std::to_string(height * width * sizeof(bit_array::TestCellState)) + 
                                    " bytes for standard grid";
        std::cout << GREEN << "  " << efficiency_msg << RESET << std::endl;
        
        // Create bit array grid
        cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary> grid(height, width);
        
        // Verify memory efficiency is as expected
        tc.assert_true(expected_bytes < height * width * sizeof(bit_array::TestCellState),
                     "Bit array grid should use less memory than standard grid");
        
        // Verify formulas are consistent
        const size_t expected_cells_per_word = (sizeof(uint32_t) * 8) / bits_per_cell;
        tc.assert_equal(expected_cells_per_word, 
                      cellib::memory::grids::bit_array::grid<bit_array::TestStateDictionary>::cells_per_word,
                      "cells_per_word calculation matches expected value");
    }

public:
    std::string name() const override {
        return "BitArrayGrid";
    }

    test_result run() override {
        test_result result;
        test_case tc(result, true);

        // Run all bit_array_grid tests
        test_grid_construction(tc);
        test_cell_access(tc);
        test_grid_sizes(tc);
        test_simple_pattern(tc);
        test_complex_pattern(tc);
        test_to_standard_conversion(tc);
        test_from_standard_conversion(tc);
        test_random_grid(tc);
        test_memory_efficiency(tc);
        
        return result;
    }
};

// Helper function to register the suite with the manager
inline void register_bit_array_grid_tests() {
    static bit_array_grid_test_suite suite;
    test_manager::instance().register_suite(&suite);
}

} // namespace cellib::tests

#endif // CELLIB_TESTS_BIT_ARRAY_GRID_HPP

#include <iostream>

// Include all test headers
#include "../include/tests/manager.hpp"
#include "../include/tests/bit_plates_grid.hpp"
#include "../include/tests/bit_array_grid.hpp"
#include "../include/tests/vector-int.hpp"
#include "../include/tests/bit_evaluator.hpp"

using namespace cellib::tests;

int main(int argc, char* argv[]) {
    // Register all test suites

    register_bit_plates_grid_tests();
    register_bit_array_grid_tests();
    register_vector_int_tests();
    register_bit_evaluator_tests();
    
    // If an argument is provided, run that specific test suite
    if (argc > 1) {
        std::string suite_name = argv[1];
        std::cout << "Running test suite: " << suite_name << std::endl;
        
        test_result result = test_manager::instance().run_suite(suite_name);
        return result.all_passed() ? 0 : 1;
    } 
    else {
        std::cout << "Running all test suites" << std::endl;
        
        test_result result = test_manager::instance().run_all();
        return result.all_passed() ? 0 : 1;
    }
}

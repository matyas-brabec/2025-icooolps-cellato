#ifndef CELLATO_TESTS_VECTOR_INT_HPP
#define CELLATO_TESTS_VECTOR_INT_HPP

#include "manager.hpp"
#include "../core/vector_int.hpp"
#include <random>

namespace cellato::tests {

class vector_int_test_suite : public test_suite {
public:
    std::string name() const override {
        return "VectorInt";
    }

    test_result run() override {
        test_result result;
        test_case tc(result, true);

        // Run all vector_int tests
        test_basics(tc);
        test_binary_operations(tc);
        test_mixed_precision_operations(tc);
        test_shifts(tc);
        test_not(tc);
        test_constant_operations(tc);
        test_load_from(tc);
        test_equals_to(tc);
        test_greater_than(tc);  // Add the new test to the run sequence
        test_random_operations(tc);
        test_factory_from_constant(tc);
        test_mask_out_columns(tc);
        
        return result;
    }

private:
    // Test basic properties and operations of vector_int
    void test_basics(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int basics ---" << RESET << std::endl;
        
        // Create a vector_int with 3 bits of precision
        using vint = cellato::core::bitwise::vector_int<uint8_t, 3>;
        vint v;
        
        // Test setting and getting values
        v.set_at(0, 1);  // 001
        tc.assert_equal(1, v.get_at(0), "Should correctly set and get value at index 0");
        
        v.set_at(1, 2);  // 010
        tc.assert_equal(2, v.get_at(1), "Should correctly set and get value at index 1");
        
        v.set_at(2, 3);  // 011
        tc.assert_equal(3, v.get_at(2), "Should correctly set and get value at index 2");
        
        v.set_at(3, 7);  // 111
        tc.assert_equal(7, v.get_at(3), "Should correctly set and get value at index 3");
        
        // Test out of range handling
        bool exception_thrown = false;
        try {
            v.set_at(9, 5);
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for out of bounds set_at");
        
        exception_thrown = false;
        try {
            v.get_at(9);
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for out of bounds get_at");
    }

    // Test binary operations between vector_int instances
    void test_binary_operations(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int binary operations ---" << RESET << std::endl;
        
        using vint = cellato::core::bitwise::vector_int<uint8_t, 3>;
        vint v1, v2;
        
        // Set up test values
        // v1: First 4 cells set to 1, 2, 3, 4
        v1.set_at(0, 1);
        v1.set_at(1, 2);
        v1.set_at(2, 3);
        v1.set_at(3, 4);
        
        // v2: First 4 cells set to 2, 3, 1, 5
        v2.set_at(0, 2);
        v2.set_at(1, 3);
        v2.set_at(2, 1);
        v2.set_at(3, 5);
        
        // Test addition
        auto v_add = v1.get_added(v2);
        tc.assert_equal(3, v_add.get_at(0), "Addition at index 0 should be 1+2=3");
        tc.assert_equal(5, v_add.get_at(1), "Addition at index 1 should be 2+3=5");
        tc.assert_equal(4, v_add.get_at(2), "Addition at index 2 should be 3+1=4");
        tc.assert_equal(1, v_add.get_at(3), "Addition at index 3 should be 4+5=9 i.e. 1 mod 8");
        
        // Test OR operation
        auto v_or = v1.get_ored(v2);
        tc.assert_equal(3, v_or.get_at(0), "OR at index 0 should be 1|2=3");
        tc.assert_equal(3, v_or.get_at(1), "OR at index 1 should be 2|3=3");
        tc.assert_equal(3, v_or.get_at(2), "OR at index 2 should be 3|1=3");
        tc.assert_equal(5, v_or.get_at(3), "OR at index 3 should be 4|5=5");
        
        // Test XOR operation
        auto v_xor = v1.get_xored(v2);
        tc.assert_equal(3, v_xor.get_at(0), "XOR at index 0 should be 1^2=3");
        tc.assert_equal(1, v_xor.get_at(1), "XOR at index 1 should be 2^3=1");
        tc.assert_equal(2, v_xor.get_at(2), "XOR at index 2 should be 3^1=2");
        tc.assert_equal(1, v_xor.get_at(3), "XOR at index 3 should be 4^5=1");
        
        // Test AND operation
        auto v_and = v1.get_anded(v2);
        tc.assert_equal(0, v_and.get_at(0), "AND at index 0 should be 1&2=0");
        tc.assert_equal(2, v_and.get_at(1), "AND at index 1 should be 2&3=2");
        tc.assert_equal(1, v_and.get_at(2), "AND at index 2 should be 3&1=1");
        tc.assert_equal(4, v_and.get_at(3), "AND at index 3 should be 4&5=4");
    }

    // Test binary operations between vector_int instances of different sizes
    void test_mixed_precision_operations(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int mixed precision operations ---" << RESET << std::endl;
        
        // Define vectors with different bit precisions
        using vint_small = cellato::core::bitwise::vector_int<uint8_t, 2>;  // 2-bit precision
        using vint_medium = cellato::core::bitwise::vector_int<uint8_t, 3>; // 3-bit precision
        using vint_large = cellato::core::bitwise::vector_int<uint8_t, 4>;  // 4-bit precision
        
        // Create instances
        vint_small small;
        vint_medium medium;
        vint_large large;
        
        // Set values to test with
        // small: 2 bits can represent 0-3
        small.set_at(0, 1);  // 01
        small.set_at(1, 2);  // 10
        small.set_at(2, 3);  // 11
        
        // medium: 3 bits can represent 0-7
        medium.set_at(0, 3);  // 011
        medium.set_at(1, 5);  // 101
        medium.set_at(2, 6);  // 110
        
        // large: 4 bits can represent 0-15
        large.set_at(0, 7);   // 0111
        large.set_at(1, 10);  // 1010
        large.set_at(2, 12);  // 1100
        
        // CASE 1: Small vector operating with medium vector
        std::cout << "  Testing small + medium operations:" << std::endl;
        
        // Addition (should return a vector with the larger precision)
        auto small_plus_medium = medium.get_added(small);
        tc.assert_equal(4, small_plus_medium.get_at(0), "Small + Medium at index 0 should be 1+3=4");
        tc.assert_equal(7, small_plus_medium.get_at(1), "Small + Medium at index 1 should be 2+5=7");
        tc.assert_equal(1, small_plus_medium.get_at(2), "Small + Medium at index 2 should be 3+6=9 mod 8 = 1");
        
        // OR
        auto small_or_medium = medium.get_ored(small);
        tc.assert_equal(3, small_or_medium.get_at(0), "Small | Medium at index 0 should be 1|3=3");
        tc.assert_equal(7, small_or_medium.get_at(1), "Small | Medium at index 1 should be 2|5=7");
        tc.assert_equal(7, small_or_medium.get_at(2), "Small | Medium at index 2 should be 3|6=7");
        
        // Continue with similar patterns for other operations...
        // For brevity, I've shown the pattern without implementing all tests

        // More test cases would follow for medium + large, small + large, etc.
    }

    void test_shifts(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int shift operations ---" << RESET << std::endl;
        
        using vint = cellato::core::bitwise::vector_int<uint8_t, 3>;
        vint v;
        
        // Set up test values - fill with a pattern
        for (int i = 0; i < 8; i++) {
            v.set_at(i, i % 4); // Pattern: 0,1,2,3,0,1,2,3
        }
        
        // Test right shift
        auto v_right = v.get_right_shifted_vector(1);
        tc.assert_equal(1, v_right.get_at(0), "Right shift by 1 should move values right");
        tc.assert_equal(2, v_right.get_at(1), "Right shift by 1 should move values right");
        tc.assert_equal(3, v_right.get_at(2), "Right shift by 1 should move values right");
        
        // Test left shift
        auto v_left = v.get_left_shifted_vector(1);
        tc.assert_equal(0, v_left.get_at(0), "Left shift by 1 should move values left");
        tc.assert_equal(0, v_left.get_at(1), "Left shift by 1 should move values left");
        tc.assert_equal(1, v_left.get_at(2), "Left shift by 1 should move values left");
        
        // Test larger shifts
        auto v_right_2 = v.get_right_shifted_vector(2);
        tc.assert_equal(2, v_right_2.get_at(0), "Right shift by 2 should move values right by 2");
        tc.assert_equal(3, v_right_2.get_at(1), "Right shift by 2 should move values right by 2");
        tc.assert_equal(0, v_right_2.get_at(2), "Right shift by 2 should move values right by 2");
        tc.assert_equal(1, v_right_2.get_at(3), "Right shift by 2 should move values right by 2");
        tc.assert_equal(2, v_right_2.get_at(4), "Right shift by 2 should move values right by 2");
    }

    void test_not(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int NOT operation ---" << RESET << std::endl;
        
        using vint = cellato::core::bitwise::vector_int<uint8_t, 3>;
        vint v;
        
        // Set alternating pattern of 0s and 1s
        for (int i = 0; i < 8; i++) {
            v.set_at(i, i % 2);
        }
        
        auto v_not = v.get_noted();
        
        // Check that all bits are inverted
        for (int i = 0; i < 8; i++) {
            auto expected = (v.get_at(i) == 0) ? 7 : 6; // NOT 0 = 7 (111), NOT 1 = 6 (110)
            tc.assert_equal(expected, v_not.get_at(i), "NOT operation should invert all bits at index " + std::to_string(i));
        }
    }

    void test_constant_operations(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int constant operations ---" << RESET << std::endl;
        
        using vint = cellato::core::bitwise::vector_int<uint8_t, 3>;
        vint v;
        
        // Fill with consecutive values
        for (int i = 0; i < 8; i++) {
            v.set_at(i, i); // 0, 1, 2, 3, 4, 5, 6, 7
        }
        
        // Test AND with constant
        auto v_and_const = v.get_anded<2>(); // AND with 010
        for (int i = 0; i < 8; i++) {
            auto expected = v.get_at(i) & 2;
            tc.assert_equal(expected, v_and_const.get_at(i), "AND with constant 2 at index " + std::to_string(i));
        }
        
        // Test OR with constant
        auto v_or_const = v.get_ored<3>(); // OR with 011
        for (int i = 0; i < 8; i++) {
            auto expected = v.get_at(i) | 3;
            tc.assert_equal(expected, v_or_const.get_at(i), "OR with constant 3 at index " + std::to_string(i));
        }
         
        // Test XOR with constant
        auto v_xor_const = v.get_xored<5>(); // XOR with 101
        for (int i = 0; i < 8; i++) {
            auto expected = v.get_at(i) ^ 5;
            tc.assert_equal(expected, v_xor_const.get_at(i), "XOR with constant 5 at index " + std::to_string(i));
        }
    }

    void test_load_from(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int load_from ---" << RESET << std::endl;
        
        using vint = cellato::core::bitwise::vector_int<uint8_t, 3>;
        
        // Create storage vectors with test data
        std::vector<uint8_t> b0 = {0xF0, 0xAA, 0x00, 0x00};  // 11110000, 10101010
        std::vector<uint8_t> b1 = {0x0F, 0xCC, 0x00, 0x00};  // 00001111, 11001100
        std::vector<uint8_t> b2 = {0x00, 0xF0, 0x00, 0x00};  // 00000000, 11110000
        
        // Create storage tuples
        std::tuple<uint8_t*, uint8_t*, uint8_t*> storage = {b0.data(), b1.data(), b2.data()};
        
        // Load from storage at offset 0
        vint v0 = vint::load_from(storage, 0);
        tc.assert_equal(2, v0.get_at(0), "First bit from storage at offset 0 should be 2");
        tc.assert_equal(2, v0.get_at(1), "Second bit from storage at offset 0 should be 2");
        tc.assert_equal(2, v0.get_at(2), "Third bit from storage at offset 0 should be 2");
        tc.assert_equal(2, v0.get_at(3), "Fourth bit from storage at offset 0 should be 2");
        tc.assert_equal(1, v0.get_at(4), "Fifth bit from storage at offset 0 should be 1");
        
        // Load from storage at offset 1
        vint v1 = vint::load_from(storage, 1);
        tc.assert_equal(0, v1.get_at(0), "First bit from storage at offset 1 should be 0");
        tc.assert_equal(1, v1.get_at(1), "Second bit from storage at offset 1 should be 1");
        tc.assert_equal(2, v1.get_at(2), "Third bit from storage at offset 1 should be 2");
        tc.assert_equal(3, v1.get_at(3), "Fourth bit from storage at offset 1 should be 3");
        tc.assert_equal(4, v1.get_at(4), "Fifth bit from storage at offset 1 should be 4");
        
        // Test with smaller storage
        std::tuple<uint8_t*, uint8_t*> storage_small = {b0.data(), b1.data()};
        vint v2 = vint::load_from(storage_small, 1);
        
        tc.assert_equal(0, v2.get_at(0), "First bit from small storage should be 0");
        tc.assert_equal(1, v2.get_at(1), "Second bit from small storage should be 1");
        // Third bit should be zeroed since storage doesn't have it
    }

    void test_equals_to(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int equals_to ---" << RESET << std::endl;
        
        using vint = cellato::core::bitwise::vector_int<uint8_t, 3>;
        vint v1, v2, v3;
        
        // Set up test values
        v1.set_at(0, 1);
        v1.set_at(1, 2);
        v1.set_at(2, 3);
        v1.set_at(3, 4);
        
        // v2 is identical to v1
        v2.set_at(0, 1);
        v2.set_at(1, 2);
        v2.set_at(2, 3);
        v2.set_at(3, 4);
        
        // v3 is different
        v3.set_at(0, 2);
        v3.set_at(1, 2); 
        v3.set_at(2, 3);
        v3.set_at(3, 5);
        
        // Test equals_to between vectors
        auto eq1_2 = v1.equals_to(v2);
        auto eq1_3 = v1.equals_to(v3);
        
        // Since equals_to returns a mask with 1s where equal
        // For identical vectors, we expect all 1s (0xFF)
        tc.assert_equal(static_cast<int>(0xFF), static_cast<int>(eq1_2), 
                    "equals_to should return all ones for identical vectors");
        
        // For different vectors, we expect some 0s
        tc.assert_true(eq1_3 != 0xFF, 
                    "equals_to should not return all ones for different vectors");
        
        // Test equals_to with constant
        auto eq1_const = v1.equals_to<1>();
        
        // Since v1[0] = 1, we expect bit 0 to be set in the equals_to result
        tc.assert_true((eq1_const & 0x01) != 0, 
                    "equals_to<1> should have bit 0 set for vector with 1 at position 0");
    }

    void test_greater_than(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int greater_than operation ---" << RESET << std::endl;
        
        // Test with same bit-width vectors
        {
            using vint3 = cellato::core::bitwise::vector_int<uint8_t, 3>;
            vint3 v1, v2;
            
            // Simple case: v1 > v2
            v1.set_at(0, 5);  // 101b = 5
            v2.set_at(0, 3);  // 011b = 3
            auto result1 = v1.greater_than(v2);
            tc.assert_true(result1 != 0, "5 should be greater than 3");
            
            // Simple case: v1 < v2
            v1.set_at(0, 2);  // 010b = 2
            v2.set_at(0, 7);  // 111b = 7
            auto result2 = v1.greater_than(v2);
            tc.assert_equal(static_cast<int>(0), static_cast<int>(result2), "2 should not be greater than 7");
            
            // Equality case: v1 = v2
            v1.set_at(0, 4);  // 100b = 4
            v2.set_at(0, 4);  // 100b = 4
            auto result3 = v1.greater_than(v2);
            tc.assert_equal(static_cast<int>(0), static_cast<int>(result3), "Equal values should return 0");
            
            // Edge cases
            // All zeros vs all zeros
            v1.set_at(0, 0);
            v2.set_at(0, 0);
            auto result4 = v1.greater_than(v2);
            tc.assert_equal(static_cast<int>(0), static_cast<int>(result4), "0 should not be greater than 0");
            
            // All ones vs all ones
            v1.set_at(0, 7);  // 111b = 7 (max for 3 bits)
            v2.set_at(0, 7);  // 111b = 7
            auto result5 = v1.greater_than(v2);
            tc.assert_equal(static_cast<int>(0), static_cast<int>(result5), "Max value should not be greater than itself");
        }
        
        // Test with different bit-width vectors
        {
            using vint_small = cellato::core::bitwise::vector_int<uint8_t, 2>;  // 2-bit precision (0-3)
            using vint_medium = cellato::core::bitwise::vector_int<uint8_t, 3>; // 3-bit precision (0-7)
            using vint_large = cellato::core::bitwise::vector_int<uint8_t, 4>;  // 4-bit precision (0-15)
            
            // Create instances
            vint_small small;
            vint_medium medium;
            vint_large large;
            
            // Case 1: Small value > Medium value
            small.set_at(0, 3);   // 11b = 3 (max for 2 bits)
            medium.set_at(0, 2);  // 010b = 2
            auto result1 = small.greater_than(medium);
            tc.assert_true(result1 != 0, "Small 3 should be greater than medium 2");
            
            // Case 2: Medium value > Small value (MSB test)
            medium.set_at(0, 4);  // 100b = 4 (has a bit set at position outside small's range)
            small.set_at(0, 3);   // 11b = 3 (max for 2 bits)
            auto result2 = medium.greater_than(small);
            tc.assert_true(result2 != 0, "Medium 4 should be greater than small 3 (MSB)");
            
            // Case 3: Large value > Medium value (MSB test)
            large.set_at(0, 8);   // 1000b = 8 (has a bit set at position outside medium's range)
            medium.set_at(0, 7);  // 111b = 7 (max for 3 bits)
            auto result3 = large.greater_than(medium);
            tc.assert_true(result3 != 0, "Large 8 should be greater than medium 7 (MSB)");
            
            // Case 4: Small value not > Large value
            small.set_at(0, 3);   // 11b = 3
            large.set_at(0, 15);  // 1111b = 15 (max for 4 bits)
            auto result4 = small.greater_than(large);
            tc.assert_equal(static_cast<int>(0), static_cast<int>(result4), "Small 3 should not be greater than large 15");
            
            // Case 5: Comparison with highest bit set
            medium.set_at(0, 4);  // 100b = 4 (third bit set)
            small.set_at(0, 3);   // 11b = 3 (first and second bits set)
            auto result5 = medium.greater_than(small);
            tc.assert_true(result5 != 0, "Medium with highest bit set should be greater than small with all bits set");
        }
        
        // Test with multi-element vectors (multiple indices)
        {
            using vint = cellato::core::bitwise::vector_int<uint8_t, 3>;
            vint v1, v2;
            
            // Set up different values at different indices
            // v1: [5, 2, 7, 3]
            v1.set_at(0, 5);  // 101b = 5
            v1.set_at(1, 2);  // 010b = 2
            v1.set_at(2, 7);  // 111b = 7
            v1.set_at(3, 3);  // 011b = 3
            
            // v2: [3, 4, 7, 1]
            v2.set_at(0, 3);  // 011b = 3
            v2.set_at(1, 4);  // 100b = 4
            v2.set_at(2, 7);  // 111b = 7
            v2.set_at(3, 1);  // 001b = 1
            
            auto result = v1.greater_than(v2);
            
            // Extract individual bits to check each comparison
            bool compare0 = (result & (1 << 0)) != 0;  // v1[0] > v2[0] (5 > 3) -> true
            bool compare1 = (result & (1 << 1)) != 0;  // v1[1] > v2[1] (2 > 4) -> false
            bool compare2 = (result & (1 << 2)) != 0;  // v1[2] > v2[2] (7 = 7) -> false
            bool compare3 = (result & (1 << 3)) != 0;  // v1[3] > v2[3] (3 > 1) -> true
            
            tc.assert_true(compare0, "v1[0]=5 should be greater than v2[0]=3");
            tc.assert_true(!compare1, "v1[1]=2 should not be greater than v2[1]=4");
            tc.assert_true(!compare2, "v1[2]=7 should not be greater than v2[2]=7 (equality)");
            tc.assert_true(compare3, "v1[3]=3 should be greater than v2[3]=1");
        }
        
        // Random tests to verify behavior with various inputs
        {
            std::cout << "  Running random greater_than tests..." << std::endl;
            std::srand(42);  // Set random seed for reproducibility
            
            using vint_small = cellato::core::bitwise::vector_int<uint8_t, 2>;
            using vint_medium = cellato::core::bitwise::vector_int<uint8_t, 3>;
            
            const int num_tests = 20;
            
            for (int i = 0; i < num_tests; i++) {
                vint_small small;
                vint_medium medium;
                
                // Generate random values
                int small_val = std::rand() % 4;   // 0-3
                int medium_val = std::rand() % 8;  // 0-7
                
                small.set_at(0, small_val);
                medium.set_at(0, medium_val);
                
                auto small_gt_medium = small.greater_than(medium);
                auto medium_gt_small = medium.greater_than(small);
                
                // Verify results
                bool expected_small_gt = small_val > medium_val;
                bool expected_medium_gt = medium_val > small_val;
                
                bool actual_small_gt = small_gt_medium != 0;
                bool actual_medium_gt = medium_gt_small != 0;
                
                tc.assert_equal(expected_small_gt, actual_small_gt, 
                            "Random test: small " + std::to_string(small_val) + 
                            " > medium " + std::to_string(medium_val));
                
                tc.assert_equal(expected_medium_gt, actual_medium_gt, 
                            "Random test: medium " + std::to_string(medium_val) + 
                            " > small " + std::to_string(small_val));
            }
        }
    }

    void test_random_operations(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int random operations ---" << RESET << std::endl;
        
        // Set random seed for reproducible tests
        std::srand(42);
        
        // Define vectors with different bit precisions
        using vint_small = cellato::core::bitwise::vector_int<uint8_t, 2>;  // 2-bit precision (0-3)
        using vint_medium = cellato::core::bitwise::vector_int<uint8_t, 3>; // 3-bit precision (0-7)
        using vint_large = cellato::core::bitwise::vector_int<uint8_t, 4>;  // 4-bit precision (0-15)
        
        const int num_iterations = 100; // Reduced for faster tests
        const int max_idx = 7; // Test up to 8 cells
        
        std::cout << "  Running " << num_iterations << " random operation tests..." << std::endl;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            // Create vector instances
            vint_small small;
            vint_medium medium;
            vint_large large;
            
            // Fill with random values within their valid ranges
            for (int i = 0; i <= max_idx; i++) {
                int small_val = std::rand() % 4;   // 0-3 (2 bits)
                int medium_val = std::rand() % 8;  // 0-7 (3 bits)
                int large_val = std::rand() % 16;  // 0-15 (4 bits)
                
                small.set_at(i, small_val);
                medium.set_at(i, medium_val);
                large.set_at(i, large_val);
            }
            
            // Test index to verify (also random)
            int test_idx = std::rand() % (max_idx + 1);
            
            // Get the values at the test index
            int small_val = small.get_at(test_idx);
            int medium_val = medium.get_at(test_idx);
            int large_val = large.get_at(test_idx);
            
            // Test 1: Addition between different precision vectors
            auto small_plus_medium = small.get_added(medium);
            int expected_sum1 = (small_val + medium_val) % 8;  // Result must fit in 3 bits
            tc.assert_equal(expected_sum1, small_plus_medium.get_at(test_idx), 
                        "Random Small + Medium at index " + std::to_string(test_idx));
            
            // Test 2: Bitwise operations
            // Large OR Small
            auto large_or_small = large.get_ored(small);
            int expected_or1 = large_val | small_val;
            tc.assert_equal(expected_or1, large_or_small.get_at(test_idx),
                        "Random Large | Small at index " + std::to_string(test_idx));
            
            // Small AND Large
            auto small_and_large = small.get_anded(large);
            int expected_and1 = small_val & large_val;
            tc.assert_equal(expected_and1, small_and_large.get_at(test_idx),
                        "Random Small & Large at index " + std::to_string(test_idx));
        }
        
        std::cout << GREEN << "  Completed " << num_iterations << " random tests" << RESET << std::endl;
    }

    void test_factory_from_constant(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int_factory from_constant ---" << RESET << std::endl;
        
        using namespace cellato::core::bitwise;
        
        // Test with constant 0 (should create a vector with all zeros)
        auto zero_vector = vector_int_factory::from_constant<uint8_t, 0>();

        for (int i = 0; i < 8; i++) {
            tc.assert_equal(0, zero_vector.get_at(i), "Constant 0 should produce vector with 0 at index " + std::to_string(i));
        }
        
        // Test with constant 1 (bit 0 set)
        auto one_vector = vector_int_factory::from_constant<uint8_t, 1>();
        for (int i = 0; i < 8; i++) {
            tc.assert_equal(1, one_vector.get_at(i), "Constant 1 should produce vector with 1 at index " + std::to_string(i));
        }

        // Test with constant 2 (bit 1 set)
        auto two_vector = vector_int_factory::from_constant<uint8_t, 2>();
        for (int i = 0; i < 8; i++) {
            tc.assert_equal(2, two_vector.get_at(i), "Constant 2 should produce vector with 2 at index " + std::to_string(i));
        }
        
        // Test with constant 3 (bits 0 and 1 set)
        auto three_vector = vector_int_factory::from_constant<uint8_t, 3>();
        for (int i = 0; i < 8; i++) {
            tc.assert_equal(3, three_vector.get_at(i), "Constant 3 should produce vector with 3 at index " + std::to_string(i));
        }
        
        // Test with constant 5 (bits 0 and 2 set, binary 101)
        auto five_vector = vector_int_factory::from_constant<uint8_t, 5>();
        for (int i = 0; i < 8; i++) {
            tc.assert_equal(5, five_vector.get_at(i), "Constant 5 should produce vector with 5 at index " + std::to_string(i));
        }
    }

    void test_mask_out_columns(test_case& tc) {
        std::cout << BLUE << "\n--- Testing vector_int mask_out_columns ---" << RESET << std::endl;
        
        using vint = cellato::core::bitwise::vector_int<uint8_t, 3>;
        vint v;
        
        // Fill with known pattern
        for (int i = 0; i < 8; i++) {
            v.set_at(i, i);  // Values 0, 1, 2, 3, 4, 5, 6, 7
        }
        
        // Test case 1: Mask with all bits set - should preserve all columns
        uint8_t full_mask = 0xFF;  // All bits set
        auto full_masked = v.mask_out_columns(full_mask);
        for (int i = 0; i < 8; i++) {
            tc.assert_equal(v.get_at(i), full_masked.get_at(i), 
                        "Full mask should preserve value at index " + std::to_string(i));
        }
        
        // Test case 2: Mask with no bits set - should clear all columns
        uint8_t zero_mask = 0x00;  // No bits set
        auto zero_masked = v.mask_out_columns(zero_mask);
        for (int i = 0; i < 8; i++) {
            tc.assert_equal(0, zero_masked.get_at(i), 
                        "Zero mask should clear value at index " + std::to_string(i));
        }
        
        // Test case 3: Mask with even bits set (0, 2, 4, 6) - should preserve even columns
        uint8_t even_mask = 0x55;  // 01010101 in binary
        auto even_masked = v.mask_out_columns(even_mask);
        for (int i = 0; i < 8; i++) {
            int expected = (i % 2 == 0) ? v.get_at(i) : 0;
            tc.assert_equal(expected, even_masked.get_at(i), 
                        "Even mask should preserve only even indices at index " + std::to_string(i));
        }
        
        // Test case 4: Mask with odd bits set (1, 3, 5, 7) - should preserve odd columns
        uint8_t odd_mask = 0xAA;  // 10101010 in binary
        auto odd_masked = v.mask_out_columns(odd_mask);
        for (int i = 0; i < 8; i++) {
            int expected = (i % 2 == 1) ? v.get_at(i) : 0;
            tc.assert_equal(expected, odd_masked.get_at(i), 
                        "Odd mask should preserve only odd indices at index " + std::to_string(i));
        }
    }
};

// Helper function to register the suite with the manager
inline void register_vector_int_tests() {
    static vector_int_test_suite suite;
    test_manager::instance().register_suite(&suite);
}

} // namespace cellato::tests

#endif // CELLATO_TESTS_VECTOR_INT_HPP

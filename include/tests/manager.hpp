#ifndef CELLATO_TESTS_MANAGER_HPP
#define CELLATO_TESTS_MANAGER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

namespace cellato::tests {

// ANSI color codes for terminal output
const std::string RESET = "\033[0m";
const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string YELLOW = "\033[33m";
const std::string BLUE = "\033[34m";
const std::string CYAN = "\033[36m";

// Base class for test results
class test_result {
public:
    int total = 0;
    int passed = 0;
    std::vector<std::string> failed_messages;

    void add(const test_result& other) {
        total += other.total;
        passed += other.passed;
        failed_messages.insert(failed_messages.end(), 
                              other.failed_messages.begin(), 
                              other.failed_messages.end());
    }

    bool all_passed() const {
        return total == passed;
    }

    int failed() const {
        return total - passed;
    }
};

// Base class for all test suites
class test_suite {
public:
    virtual ~test_suite() = default;
    virtual std::string name() const = 0;
    virtual test_result run() = 0;
};

// Test manager singleton
class test_manager {
private:
    std::vector<test_suite*> test_suites;
    bool verbose_output = true;

    test_manager() = default;
    
public:
    // Get the singleton instance
    static test_manager& instance() {
        static test_manager instance;
        return instance;
    }

    // Register a test suite
    void register_suite(test_suite* suite) {
        test_suites.push_back(suite);
    }

    // Set verbosity
    void set_verbose(bool verbose) {
        verbose_output = verbose;
    }

    // Run all registered test suites
    test_result run_all() {
        std::cout << CYAN << "========================================" << std::endl;
        std::cout << "   RUNNING ALL TEST SUITES" << std::endl;
        std::cout << "========================================" << RESET << std::endl;

        test_result overall;
        
        for (auto* suite : test_suites) {
            std::cout << BLUE << "\n====== RUNNING TEST SUITE: " << suite->name() << " ======" << RESET << std::endl;
            
            test_result suite_result = suite->run();
            overall.add(suite_result);
            
            std::cout << BLUE << "====== COMPLETED TEST SUITE: " << suite->name() 
                     << " (" << suite_result.passed << "/" << suite_result.total << " passed) ======" << RESET << std::endl;
        }

        print_summary(overall);
        return overall;
    }

    // Run a specific test suite by name
    test_result run_suite(const std::string& suite_name) {
        for (auto* suite : test_suites) {
            if (suite->name() == suite_name) {
                std::cout << BLUE << "\n====== RUNNING TEST SUITE: " << suite->name() << " ======" << RESET << std::endl;
                
                test_result result = suite->run();
                
                std::cout << BLUE << "====== COMPLETED TEST SUITE: " << suite->name() 
                         << " (" << result.passed << "/" << result.total << " passed) ======" << RESET << std::endl;
                
                print_summary(result);
                return result;
            }
        }
        
        std::cout << RED << "Test suite '" << suite_name << "' not found!" << RESET << std::endl;
        return {};
    }

private:
    void print_summary(const test_result& result) {
        std::cout << YELLOW << "\n====== TEST SUMMARY ======" << RESET << std::endl << std::endl;
        std::cout << BLUE << "Total tests: " << result.total << std::endl;
        
        std::cout << GREEN << "  Tests passed: " << result.passed << RESET << std::endl;
        std::cout << RED << "  Tests failed: " << result.failed() << RESET << std::endl << std::endl;

        if (!result.all_passed()) {
            std::cout << RED << "Failed tests: " << result.failed() << RESET << std::endl;
            for (size_t i = 0; i < result.failed_messages.size(); ++i) {
                std::cout << RED << (i+1) << ". " << result.failed_messages[i] << RESET << std::endl;
            }
        } else {
            std::cout << GREEN << "All tests passed!" << RESET << std::endl << std::endl;
        }
    }
};

// Helper for assert functions
class test_case {
public:
    test_case(test_result& result, bool verbose = true) 
        : result(result), verbose(verbose) {}

    template<typename T, typename U>
    void assert_equal(T expected, U actual, const std::string& message) {
        result.total++;
        
        if (expected == static_cast<T>(actual)) {
            if (verbose) {
                std::cout << GREEN << "✓ PASS: " << message << RESET << std::endl;
            }
            result.passed++;
        } else {
            std::string failure_msg = message + " (Expected: " + to_string(expected) + 
                                     ", Got: " + to_string(actual) + ")";
            if (verbose) {
                std::cout << RED << "✗ FAIL: " << failure_msg << RESET << std::endl;
            }
            result.failed_messages.push_back(failure_msg);
        }
    }

    void assert_true(bool condition, const std::string& message) {
        result.total++;
        
        if (condition) {
            if (verbose) {
                std::cout << GREEN << "✓ PASS: " << message << RESET << std::endl;
            }
            result.passed++;
        } else {
            if (verbose) {
                std::cout << RED << "✗ FAIL: " << message << RESET << std::endl;
            }
            result.failed_messages.push_back(message);
        }
    }

private:
    test_result& result;
    bool verbose;

    // Helper to convert any type to string
    template<typename T>
    std::string to_string(const T& value) {
        return std::to_string(value);
    }
};

} // namespace cellato::tests

#endif // CELLATO_TESTS_MANAGER_HPP

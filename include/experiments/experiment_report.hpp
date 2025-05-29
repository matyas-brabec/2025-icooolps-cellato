#ifndef CELLATO_EXPERIMENT_REPORT_HPP
#define CELLATO_EXPERIMENT_REPORT_HPP

#include "./run_params.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

namespace cellato::run {

struct experiment_report {
    run_params params;

    std::vector<double> execution_times_ms;
    std::vector<std::string> checksums;

    void pretty_print(std::ostream& os) const {
        // Define ANSI color codes for colorful output
        const std::string RESET = "\033[0m";
        const std::string BOLD = "\033[1m";
        const std::string BLUE = "\033[34m";
        const std::string GREEN = "\033[32m";
        const std::string YELLOW = "\033[33m";
        const std::string CYAN = "\033[36m";
        const std::string RED = "\033[31m";
        
        // Print header and basic information
        os << BOLD << BLUE << "==============================================" << RESET << std::endl;
        os << BOLD << BLUE << "          EXPERIMENT REPORT                  " << RESET << std::endl;
        os << BOLD << BLUE << "==============================================" << RESET << std::endl;
        
        // Print experiment parameters
        os << BOLD << "Experiment Parameters:" << RESET << std::endl;
        os << "  " << CYAN << "Grid Size: " << RESET << params.x_size << " x " << params.y_size 
           << " (" << params.x_size * params.y_size << " cells)" << std::endl;
        os << "  " << CYAN << "Steps: " << RESET << params.steps << std::endl;
        if (params.seed != -1) {
            os << "  " << CYAN << "Random Seed: " << RESET << params.seed << std::endl;
        }
        
        // Print performance metrics
        os << std::endl << BOLD << "Performance Metrics:" << RESET << std::endl;
        os << "  " << YELLOW << "Avg. Execution Time: " << RESET << average_time_ms() << " ms" << std::endl;
        os << "  " << YELLOW << "Std. Deviation: " << RESET << std_time_ms() << " ms" << std::endl;
        os << "  " << YELLOW << "Avg. Time Per Cell: " << RESET << average_time_per_cell_ns() << " ns" << std::endl;
        
        // Print number of executions
        os << "  " << YELLOW << "Number of Executions: " << RESET << execution_times_ms.size() << std::endl;
        
        // Print detailed time information if we have multiple executions
        if (execution_times_ms.size() > 1) {
            os << std::endl << BOLD << "Execution Times (ms):" << RESET << std::endl << "  ";
            for (size_t i = 0; i < execution_times_ms.size(); ++i) {
                os << execution_times_ms[i];
                if (i < execution_times_ms.size() - 1) {
                    os << ", ";
                }
                // Add line breaks for readability
                if ((i + 1) % 5 == 0 && i < execution_times_ms.size() - 1) {
                    os << std::endl << "  ";
                }
            }
            os << std::endl;
        }
        
        // Print checksum information
        os << std::endl << BOLD << "Consistency Check:" << RESET << std::endl;
        if (rounds_had_same_checksums()) {
            os << "  " << GREEN << "All rounds produced identical results!" << RESET << std::endl;
        } else {
            os << "  " << RED << "Warning: Different rounds produced different results!" << RESET << std::endl;
        }
        
        if (!checksums.empty()) {
            os << "  " << CYAN << "Checksum: " << RESET << checksums[0] << std::endl;
        }
        
        os << BOLD << BLUE << "==============================================" << RESET << std::endl;
    }

    static std::string csv_header() {
        return run_params::csv_header() + ",average_time_ms,average_time_per_cell_ns,std_time_ms,rounds_had_same_checksums,checksum";
    }

    std::string csv_line() const {
        return params.csv_line() + "," +
               std::to_string(average_time_ms()) + "," +
               std::to_string(average_time_per_cell_ns()) + "," +
               std::to_string(std_time_ms()) + "," +
               (rounds_had_same_checksums() ? "true" : "false") + "," +
               checksums[0];
    }

    double average_time_ms() const {
        double sum = 0;
        for (const auto& time : execution_times_ms) {
            sum += time;
        }
        return sum / execution_times_ms.size();
    }

    double std_time_ms() const {
        double mean = average_time_ms();
        double sum = 0;
        for (const auto& time : execution_times_ms) {
            sum += (time - mean) * (time - mean);
        }
        return std::sqrt(sum / execution_times_ms.size());
    }

    bool rounds_had_same_checksums() const {
        std::size_t count = 0;
        for (const auto& checksum : checksums) {
            if (checksum == checksums[0]) {
                count++;
            }
        }
        return count == checksums.size();
    }

    double average_time_per_cell_ns() const {
        if (params.x_size == 0 || params.y_size == 0) {
            return 0;
        }
        return (average_time_ms() * 1e6) / (params.x_size * params.y_size);
    }
};

}

#endif // CELLATO_EXPERIMENT_REPORT_HPP
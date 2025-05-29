#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <string>

#include "experiments/run_params.hpp"
#include "experiments/test_suites.hpp"
#include "experiments/experiment_manager.hpp"
#include "experiments/reference_impl_manager.hpp"
#include "memory/grid_utils.hpp"

#include "game_of_life/algorithm.hpp"
#include "game_of_life/pretty_print.hpp"
#include "game_of_life/config.hpp"
#include "fire/config.hpp"
#include "greenberg/config.hpp"
#include "wire/config.hpp"

#include "args-parser.hpp"
#include "_relwork/runner_wrapper.hpp"
#include "_relwork/__framework_of_choice__/fire/runner.hpp"
#include "_relwork/__framework_of_choice__/game_of_life/runner.hpp"


#ifdef ENABLE_KOKKOS
#include "_relwork/kokkos/game_of_life/runner.hpp"
#include "_relwork/kokkos/fire/runner.hpp"
#include "_relwork/kokkos/greenberg/runner.hpp"
#include "_relwork/kokkos/wire/runner.hpp"
#endif // ENABLE_KOKKOS

#ifdef ENABLE_GRIDTOOLS
#include "_relwork/gridtools/game_of_life/runner.hpp" 
// #include "_relwork/gridtools/fire/runner.hpp"
// #include "_relwork/gridtools/greenberg/runner.hpp"
// #include "_relwork/gridtools/wire/runner.hpp"
#endif // ENABLE_GRIDTOOLS

#ifdef ENABLE_HALIDE
#include "_relwork/halide/game_of_life/runner.hpp"
// #include "_relwork/halide/fire/runner.hpp"
// #include "_relwork/halide/greenberg/runner.hpp"
// #include "_relwork/halide/wire/runner.hpp"
#endif // ENABLE_HALIDE


#define LOG std::cerr
#define REPORT std::cout

template <typename... all_test_suites>
struct switch_ {
    static void run(cellato::run::run_params& params) {
        // If reference implementation is requested, handle it separately
        if (params.reference_impl != "none") {
            bool ref_executed = run_reference_impl(params);
            if (!ref_executed) {
                std::cerr << "No suitable reference implementation found for the given parameters." << std::endl;
            }
            return;
        }
        
        bool any_executed = (call<all_test_suites>(params) || ...);
        if (!any_executed) {
            std::cerr << "No suitable test suite found for the given parameters." << std::endl;
        }
    }

private:
    static bool run_reference_impl(cellato::run::run_params& params) {

        if (params.reference_impl == "baseline") {
            if (params.automaton == "game-of-life") {
                return run_reference_for_automaton<game_of_life::config>(params);
            } else if (params.automaton == "fire" || params.automaton == "forest-fire") {
                return run_reference_for_automaton<fire::config>(params);
            } else if (params.automaton == "greenberg-hastings") {
                return run_reference_for_automaton<greenberg::config>(params);
            } else if (params.automaton == "wire") {
                return run_reference_for_automaton<wire::config>(params);
            }
        }

        else if (params.reference_impl == "_framework_of_choice_") {
            if (params.automaton == "game-of-life") {
                return run_relwork<game_of_life::config, __framework_of_choice__::game_of_life::runner>(params);
            } else if (params.automaton == "fire" || params.automaton == "forest-fire") {
                return run_relwork<fire::config, __framework_of_choice__::fire::runner>(params);
            }
            // ...
        }

#ifdef ENABLE_GRIDTOOLS
        else if (params.reference_impl == "gridtools") {
            if (params.automaton == "game-of-life") {
                return run_relwork<game_of_life::config, gridtools::game_of_life::runner>(params);
            } else if (params.automaton == "fire" || params.automaton == "forest-fire") {
                // return run_relwork<fire::config, gridtools::fire::runner>(params);
            } else if (params.automaton == "greenberg-hastings") {
                // return run_relwork<greenberg::config, gridtools::greenberg::runner>(params);
            } else if (params.automaton == "wire") {
                // return run_relwork<wire::config, gridtools::wire::runner>(params);
            }
        }
#else
        else if (params.reference_impl == "gridtools") {
            std::cerr << "GridTools reference implementation is not enabled in this build." << std::endl;
            return false;
        }
#endif // ENABLE_GRIDTOOLS

#ifdef ENABLE_KOKKOS
        else if (params.reference_impl == "kokkos") {
            if (params.automaton == "game-of-life") {
                return run_relwork<game_of_life::config, kokkos::game_of_life::runner>(params);
            } else if (params.automaton == "fire" || params.automaton == "forest-fire") {
                return run_relwork<fire::config, kokkos::fire::runner>(params);
            } else if (params.automaton == "greenberg-hastings") {
                return run_relwork<greenberg::config, kokkos::greenberg::runner>(params);
            } else if (params.automaton == "wire") {
                return run_relwork<wire::config, kokkos::wire::runner>(params);
            }
        }
#else
        else if (params.reference_impl == "kokkos") {
            std::cerr << "Kokkos reference implementation is not enabled in this build." << std::endl;
            return false;
        }
#endif // ENABLE_HALIDE

#ifdef ENABLE_HALIDE
        else if (params.reference_impl == "halide") {
            if (params.automaton == "game-of-life") {
                return run_relwork<game_of_life::config, halide::game_of_life::runner>(params);
            } else if (params.automaton == "fire" || params.automaton == "forest-fire") {
                // return run_relwork<fire::config, halide::fire::runner>(params);
            } else if (params.automaton == "greenberg-hastings") {
                // return run_relwork<greenberg::config, halide::greenberg::runner>(params);
            } else if (params.automaton == "wire") {
                // return run_relwork<wire::config, halide::wire::runner>(params);
            }
        }
#else
        else if (params.reference_impl == "halide") {
            std::cerr << "Halide reference implementation is not enabled in this build." << std::endl;
            return false;
        }
#endif // ENABLE_HALIDE

        return false;
    }

    template <typename automaton_config, typename relwork_runner>
    static bool run_relwork(cellato::run::run_params& params) {
        return run_reference_for_automaton<
            automaton_config,
            relwork::runner_wrapper<
                automaton_config, 
                relwork_runner>>(params);
    }

    template <typename automaton_config,
              typename runner_t = typename automaton_config::reference_implementation>
    static bool run_reference_for_automaton(cellato::run::run_params& params) {
        using cell_state_t = typename automaton_config::cell_state;
        
        // Generate initial state using the automaton's random initializer
        auto initial_state = automaton_config::input::random::init(params);
        
        // Run the reference implementation
        cellato::run::reference_impl_manager<runner_t, cell_state_t> manager;
        auto report = manager.run_experiment(params, initial_state);
        
        REPORT << report.csv_line() << std::endl;
        report.pretty_print(LOG);
        
        return true;
    }

    template <typename test_suite>
    static bool call(cellato::run::run_params& params) {
        if (!test_suite::is_for(params)) {
            return false;
        }

        using cellular_automaton = typename test_suite::automaton;

        auto initial_state = cellular_automaton::input::random::init(params);

        cellato::run::experiment_manager<test_suite> manager;
        manager.set_print_config(cellular_automaton::pretty_print::get_config());

        auto report = manager.run_experiment(
            params, initial_state
        );

        REPORT << report.csv_line() << std::endl;
        
        report.pretty_print(LOG);

        return true;
    }
};


template <typename test_suite>
void run(cellato::run::run_params& params) {

}

cellato::run::run_params get_params(int argc, char* argv[]) {
    input::parser parser {argc, argv};

    if (parser.exists("help")) {
        return cellato::run::run_params{.help = true};
    }

    if (parser.exists("print_csv_header")) {
        return cellato::run::run_params{.print_csv_header = true};
    }

    std::vector<std::string> required {
        "automaton",
        "device", "traverser", "evaluator", "layout",
        "x_size", "y_size", "steps",
    };

    std::vector<std::string> optional {
        "print", "precision", "x_tile_size", "y_tile_size",
        "seed", "rounds", "warmup_rounds", "print_csv_header",
        "reference_impl", "cuda_block_size_x", "cuda_block_size_y"
    };

    if (parser.exists("evaluator")) {
        if (parser.get("evaluator") == "bit_plates" || parser.get("evaluator") == "bit_array") {
            required.push_back("precision");
        }
    }

    if (parser.exists("reference_impl")) {
        for (const auto& no_longer_required : {
            "device", "traverser", "evaluator", "layout"
        }) {
            required.erase(
                std::remove(required.begin(), required.end(), no_longer_required),
                required.end()
            );
        }
    }

    for (const auto& opt : required) {
        if (!parser.exists(opt)) {
            std::cerr << "Missing required option: " << opt << std::endl;
            exit(1);
        }
    }

    cellato::run::run_params params {
        .automaton = parser.get("automaton"),

        .device = parser.get("device"),
        .traverser = parser.get("traverser"),
        .evaluator = parser.get("evaluator"),
        .layout = parser.get("layout"),

        .reference_impl = parser.exists("reference_impl") ? parser.get("reference_impl") : "none",

        .x_size = std::stoi(parser.get("x_size")),
        .y_size = std::stoi(parser.get("y_size")),
        .steps = std::stoi(parser.get("steps")),

        .precision = parser.exists("precision") ? std::stoi(parser.get("precision")) : 0,
        
        .x_tile_size = parser.exists("x_tile_size") ? std::stoi(parser.get("x_tile_size")) : 0,
        .y_tile_size = parser.exists("y_tile_size") ? std::stoi(parser.get("y_tile_size")) : 0,
        
        .rounds = parser.exists("rounds") ? std::stoi(parser.get("rounds")) : 1,
        .warmup_rounds = parser.exists("warmup_rounds") ? std::stoi(parser.get("warmup_rounds")) : 0,

        .seed = parser.exists("seed") ? std::stoi(parser.get("seed")) : 42,

        .print = parser.exists("print"),
        .help = parser.exists("help"),
        .print_csv_header = parser.exists("print_csv_header"),

        .cuda_block_size_x = parser.exists("cuda_block_size_x") ? std::stoi(parser.get("cuda_block_size_x")) : 32,
        .cuda_block_size_y = parser.exists("cuda_block_size_y") ? std::stoi(parser.get("cuda_block_size_y")) : 8
    };

    return params;
}

void print_usage() {
    std::cout << "Usage: ./cellato [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --automaton <name>           Name of the cellular automaton\n";
    std::cout << "  --device <name>              Device to run on (CPU, CUDA)\n";
    std::cout << "  --traverser <name>           Traverser type (simple, spacial_blocking)\n";
    std::cout << "  --evaluator <name>           Evaluator type (standard, bit_plates)\n";
    std::cout << "  --layout <name>              Layout type (standard, bit_array, bit_plates)\n";
    std::cout << "  --reference_impl <name>      Reference implementation to use (baseline, kokkos, halide, gridtools)\n";
    std::cout << "  --x_size <number>            X size of the grid\n";
    std::cout << "  --y_size <number>            Y size of the grid\n";
    std::cout << "  --x_tile_size <number>       X tile size for CUDA\n";
    std::cout << "  --y_tile_size <number>       Y tile size for CUDA\n";
    std::cout << "  --rounds <number>            Number of rounds to run\n";
    std::cout << "  --warmup_rounds <number>     Number of warmup rounds to run\n";
    std::cout << "  --steps <number>             Number of steps to run\n";
    std::cout << "  --precision <number>         Precision for floating-point calculations (32, 64)\n";
    std::cout << "  --seed <number>              Random seed for initialization\n";
    std::cout << "  --print                      Print the grid after each step\n";
    std::cout << "  --reference_impl             Use reference implementation for the automaton\n";
    std::cout << "  --cuda_block_size_x <number> CUDA block size X (default: 16)\n";
    std::cout << "  --cuda_block_size_y <number> CUDA block size Y (default: 16)\n";
    std::cout << "  --print_csv_header           Print CSV header\n";
    std::cout << "  --help                       Show this help message\n";
}


int main(int argc, char* argv[]) {
    
    auto params = get_params(argc, argv);

    if (params.help) {
        print_usage();
        return 0;
    }

    if (params.print_csv_header) {
        std::cout << cellato::run::experiment_report::csv_header() << std::endl;
        return 0;
    }

    if (params.print) {
        params.print_std();
    }

    namespace test = cellato::run::test_suites;

    using _game_of_life_ = game_of_life::config;
    using _fire_ = fire::config;
    using _wire_ = wire::config;
    using _greenberg_ = greenberg::config;

    #define cases_for(automaton) \
        test::on_cpu::standard<automaton>, \
        test::on_cpu::using_<std::uint32_t>::bit_array<automaton>, \
        test::on_cpu::using_<std::uint64_t>::bit_array<automaton>, \
        test::on_cpu::using_<std::uint32_t>::bit_plates<automaton>, \
        test::on_cpu::using_<std::uint64_t>::bit_plates<automaton>, \
        test::on_cuda::standard<automaton>, \
        test::on_cuda::standard<automaton>::with_spacial_blocking<1, 1>, \
        test::on_cuda::standard<automaton>::with_spacial_blocking<2, 1>, \
        test::on_cuda::standard<automaton>::with_spacial_blocking<4, 1>, \
        test::on_cuda::using_<std::uint32_t>::bit_array<automaton>, \
        test::on_cuda::using_<std::uint64_t>::bit_array<automaton>, \
        test::on_cuda::using_<std::uint32_t>::bit_plates<automaton>, \
        test::on_cuda::using_<std::uint64_t>::bit_plates<automaton>

    switch_<
        cases_for(_game_of_life_),
        cases_for(_fire_),
        cases_for(_wire_),
        cases_for(_greenberg_)
    >::run(params);

    return 0;
}

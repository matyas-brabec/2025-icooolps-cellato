#ifndef CELLIB_TEST_SUITES_HPP
#define CELLIB_TEST_SUITES_HPP

#include "../evaluators/standard.hpp"
#include "../evaluators/bit_plates.hpp"
#include "../evaluators/bit_array.hpp"
#include "../memory/standard_grid.hpp"
#include "../traversers/cpu/simple.hpp"
#include "../memory/bit_plates_grid.hpp"
#include "../memory/bit_array_grid.hpp"
#include "../traversers/cuda/simple.hpp"
#include "../traversers/cuda/spacial_blocking.hpp"
#include "./run_params.hpp"
namespace cellib::run::test_suites {

namespace grids = cellib::memory::grids;
namespace evaluators = cellib::evaluators;

#define CUDA_OPT "CUDA"
#define CPU_OPT "CPU"

namespace on_cuda {
    
    namespace traversers = cellib::traversers::cuda;

    template <typename cellular_automaton>
    struct standard {
        using automaton = cellular_automaton;
        
        using original_cell_t = typename cellular_automaton::cell_state;
        using grid_store_word_t = original_cell_t;

        using algorithm_t = typename cellular_automaton::algorithm;
        
        using grid_t = grids::standard::grid<original_cell_t>;
        using evaluator_t = evaluators::standard::evaluator<original_cell_t, algorithm_t>;

        using traverser_t = traversers::simple::traverser<evaluator_t, grid_t>;

        constexpr static int x_margin = 1;
        constexpr static int y_margin = 1;

        static bool is_for(cellib::run::run_params& params) {
            return params.automaton == cellular_automaton::name &&
                   params.traverser == "simple" &&
                   params.device == CUDA_OPT &&
                   params.evaluator == "standard" &&
                   params.layout == "standard";
        }

        template <int y_tile_size, int x_tile_size = 1>
        struct with_spacial_blocking {
            using automaton = cellular_automaton;

            using original_cell_t = typename cellular_automaton::cell_state;
            using grid_store_word_t = original_cell_t;

            using algorithm_t = typename cellular_automaton::algorithm;
            
            using grid_t = grids::standard::grid<original_cell_t>;
            using evaluator_t = evaluators::standard::evaluator<original_cell_t, algorithm_t>;

            using traverser_t = traversers::simple::traverser<evaluator_t, grid_t>;

            constexpr static int x_margin = 1;
            constexpr static int y_margin = 1;

            static bool is_for(cellib::run::run_params& params) {
                return params.automaton == cellular_automaton::name &&
                       params.traverser == "spacial_blocking" &&
                       params.device == CUDA_OPT &&
                       params.evaluator == "standard" &&
                       params.layout == "standard" &&
                       params.y_tile_size == y_tile_size &&
                       params.x_tile_size == x_tile_size;
            }
        };
    };

    template <typename store_word_type>  
    struct using_ {

        template <typename cellular_automaton>
        struct bit_array {
            using automaton = cellular_automaton;

            using original_cell_t = typename cellular_automaton::cell_state;
            using grid_store_word_t = store_word_type;

            using algorithm_t = typename cellular_automaton::algorithm;
            using state_dictionary_t = typename cellular_automaton::state_dictionary;
            
            using grid_t = grids::bit_array::grid<state_dictionary_t, grid_store_word_t>;
            using evaluator_t = evaluators::bit_array::evaluator<grid_t, algorithm_t>;

            using traverser_t = traversers::spacial_blocking::traverser<
                evaluator_t, grid_t, 1, grid_t::cells_per_word>;
            
            constexpr static int x_margin = grid_t::cells_per_word;
            constexpr static int y_margin = 1;

            static bool is_for(cellib::run::run_params& params) {
                return params.automaton == cellular_automaton::name &&
                       params.traverser == "simple" &&
                       params.device == CUDA_OPT &&
                       params.evaluator == "bit_array" &&
                       params.layout == "bit_array" &&
                       params.precision == sizeof(store_word_type) * 8;
            }
        };

        template <typename cellular_automaton>
        struct bit_plates {
            using automaton = cellular_automaton;

            using original_cell_t = typename cellular_automaton::cell_state;
            using grid_store_word_t = store_word_type;

            using algorithm_t = typename cellular_automaton::algorithm;
            using state_dictionary_t = typename cellular_automaton::state_dictionary;

            using grid_t = grids::bit_plates::grid<grid_store_word_t, state_dictionary_t>;
            using evaluator_t = evaluators::bit_plates::evaluator<grid_store_word_t, state_dictionary_t, algorithm_t>; 

            using traverser_t = traversers::simple::traverser<evaluator_t, grid_t>;

            constexpr static int x_margin = sizeof(grid_store_word_t) * 8;
            constexpr static int y_margin = 1;

            static bool is_for(cellib::run::run_params& params) {
                return params.automaton == cellular_automaton::name &&
                       params.traverser == "simple" &&
                       params.device == CUDA_OPT &&
                       params.evaluator == "bit_plates" &&
                       params.layout == "bit_plates" &&
                       params.precision == sizeof(grid_store_word_t) * 8;
            }
        };
    };
}

namespace on_cpu {

    namespace traversers = cellib::traversers::cpu;

    template <typename cellular_automaton>
    struct standard {
        using automaton = cellular_automaton;
        
        using original_cell_t = typename cellular_automaton::cell_state;
        using grid_store_word_t = original_cell_t;

        using algorithm_t = typename cellular_automaton::algorithm;
        
        using grid_t = grids::standard::grid<original_cell_t>;
        using evaluator_t = evaluators::standard::evaluator<original_cell_t, algorithm_t>;
        
        using traverser_t = traversers::simple::traverser<evaluator_t, grid_t>;
        
        constexpr static int x_margin = 1;
        constexpr static int y_margin = 1;

        static bool is_for(cellib::run::run_params& params) {
            return params.automaton == cellular_automaton::name &&
                   params.traverser == "simple" &&
                   params.device == CPU_OPT &&
                   params.evaluator == "standard" &&
                   params.layout == "standard";
        }
    };

    template <typename store_word_type>  
    struct using_ {

        template <typename cellular_automaton>
        struct bit_array {
            using automaton = cellular_automaton;
            
            using original_cell_t = typename cellular_automaton::cell_state;
            using grid_store_word_t = store_word_type;

            using algorithm_t = typename cellular_automaton::algorithm;
            using state_dictionary_t = typename cellular_automaton::state_dictionary;
            
            using grid_t = grids::bit_array::grid<state_dictionary_t, grid_store_word_t>;
            using evaluator_t = evaluators::bit_array::evaluator<grid_t, algorithm_t>;

            using traverser_t = traversers::simple::traverser<evaluator_t, grid_t>;
            
            constexpr static int x_margin = grid_t::cells_per_word;
            constexpr static int y_margin = 1;

            static bool is_for(cellib::run::run_params& params) {
                return params.automaton == cellular_automaton::name &&
                       params.traverser == "simple" &&
                       params.device == CPU_OPT &&
                       params.evaluator == "bit_array" &&
                       params.layout == "bit_array" &&
                       params.precision == sizeof(store_word_type) * 8;
            }
        };

        template <typename cellular_automaton>
        struct bit_plates {
            using automaton = cellular_automaton;
            
            using original_cell_t = typename cellular_automaton::cell_state;
            using grid_store_word_t = store_word_type;

            using algorithm_t = typename cellular_automaton::algorithm;
            using state_dictionary_t = typename cellular_automaton::state_dictionary;

            using grid_t = grids::bit_plates::grid<grid_store_word_t, state_dictionary_t>;
            using evaluator_t = evaluators::bit_plates::evaluator<grid_store_word_t, state_dictionary_t, algorithm_t>; 
            
            using traverser_t = traversers::simple::traverser<evaluator_t, grid_t>;

            constexpr static int x_margin = sizeof(grid_store_word_t) * 8;
            constexpr static int y_margin = 1;

            static bool is_for(cellib::run::run_params& params) {
                return params.automaton == cellular_automaton::name &&
                       params.traverser == "simple" &&
                       params.device == CPU_OPT &&
                       params.evaluator == "bit_plates" &&
                       params.layout == "bit_plates" &&
                       params.precision == sizeof(grid_store_word_t) * 8;
            }
        };
    };
}

} // namespace cellib::run

#endif // CELLIB_TEST_SUITES_HPP
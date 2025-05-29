#ifndef AUTOMATON_NAMESPACE
static_assert(false, "AUTOMATON_NAMESPACE must be defined");
#endif

#include <cstdint>

#include "traversers/cuda/simple.hpp"
#include "evaluators/standard.hpp"
#include "evaluators/bit_plates.hpp"
#include "evaluators/bit_array.hpp"
#include "memory/standard_grid.hpp"
#include "memory/bit_plates_grid.hpp"
#include "memory/bit_array_grid.hpp"
#include "memory/interface.hpp"

#ifdef SIMPLE_CUDA_TRAVERSER_INSTANTIATIONS

// Standard grid with standard evaluator
#define TRAVERSER_TYPE \
    cellato::traversers::cuda::simple::traverser< \
        cellato::evaluators::standard::evaluator<AUTOMATON_NAMESPACE::config::cell_state, AUTOMATON_NAMESPACE::config::algorithm>, \
        cellato::memory::grids::standard::grid<AUTOMATON_NAMESPACE::config::cell_state, cellato::memory::grids::device::CPU> \
    >

template class TRAVERSER_TYPE;
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::simple::_run_mode::QUIET>(int);
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::simple::_run_mode::VERBOSE>(int);

#undef TRAVERSER_TYPE

// Bit plates grid with bit plates evaluator (32-bit)
#define TRAVERSER_TYPE \
    cellato::traversers::cuda::simple::traverser< \
        cellato::evaluators::bit_plates::evaluator<std::uint32_t, AUTOMATON_NAMESPACE::config::state_dictionary, AUTOMATON_NAMESPACE::config::algorithm>, \
        cellato::memory::grids::bit_plates::grid<std::uint32_t, AUTOMATON_NAMESPACE::config::state_dictionary, cellato::memory::grids::device::CPU> \
    >

template class TRAVERSER_TYPE;
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::simple::_run_mode::QUIET>(int);
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::simple::_run_mode::VERBOSE>(int);

#undef TRAVERSER_TYPE

// Bit plates grid with bit plates evaluator (64-bit)
#define TRAVERSER_TYPE \
    cellato::traversers::cuda::simple::traverser< \
        cellato::evaluators::bit_plates::evaluator<std::uint64_t, AUTOMATON_NAMESPACE::config::state_dictionary, AUTOMATON_NAMESPACE::config::algorithm>, \
        cellato::memory::grids::bit_plates::grid<std::uint64_t, AUTOMATON_NAMESPACE::config::state_dictionary, cellato::memory::grids::device::CPU> \
    >

template class TRAVERSER_TYPE;
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::simple::_run_mode::QUIET>(int);
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::simple::_run_mode::VERBOSE>(int);

#undef TRAVERSER_TYPE

#endif // SIMPLE_CUDA_TRAVERSER_INSTANTIATIONS
#ifdef SPACIAL_BLOCKING_CUDA_TRAVERSER_INSTANTIATIONS


#define GRID_TYPE \
    cellato::memory::grids::bit_array::grid< \
        AUTOMATON_NAMESPACE::config::state_dictionary, \
        std::uint32_t, \
        cellato::memory::grids::device::CPU \
    >
#define TRAVERSER_TYPE \
    cellato::traversers::cuda::spacial_blocking::traverser< \
        cellato::evaluators::bit_array::evaluator< \
            GRID_TYPE, AUTOMATON_NAMESPACE::config::algorithm>, \
        GRID_TYPE, \
        1, GRID_TYPE::cells_per_word \
    >

template class TRAVERSER_TYPE;
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::spacial_blocking::_run_mode::QUIET>(int);
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::spacial_blocking::_run_mode::VERBOSE>(int);

#undef GRID_TYPE
#undef TRAVERSER_TYPE

#define GRID_TYPE \
    cellato::memory::grids::bit_array::grid< \
        AUTOMATON_NAMESPACE::config::state_dictionary, \
        std::uint64_t, \
        cellato::memory::grids::device::CPU \
    >
#define TRAVERSER_TYPE \
    cellato::traversers::cuda::spacial_blocking::traverser< \
        cellato::evaluators::bit_array::evaluator< \
            GRID_TYPE, AUTOMATON_NAMESPACE::config::algorithm>, \
        GRID_TYPE, \
        1, GRID_TYPE::cells_per_word \
    >

template class TRAVERSER_TYPE;
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::spacial_blocking::_run_mode::QUIET>(int);
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::spacial_blocking::_run_mode::VERBOSE>(int);

#undef GRID_TYPE
#undef TRAVERSER_TYPE


#define Y_TILE_SIZE 1

    #define X_TILE_SIZE 1
    #include "cuda_instantiation_of_spacial_blocking.cuh"
    #undef X_TILE_SIZE

#undef Y_TILE_SIZE

#define Y_TILE_SIZE 2

    #define X_TILE_SIZE 1
    #include "cuda_instantiation_of_spacial_blocking.cuh"
    #undef X_TILE_SIZE

#undef Y_TILE_SIZE

#define Y_TILE_SIZE 4

    #define X_TILE_SIZE 1
    #include "cuda_instantiation_of_spacial_blocking.cuh"
    #undef X_TILE_SIZE

#undef Y_TILE_SIZE

#define Y_TILE_SIZE 8

    #define X_TILE_SIZE 1
    #include "cuda_instantiation_of_spacial_blocking.cuh"
    #undef X_TILE_SIZE

#undef Y_TILE_SIZE

#endif // SPACIAL_BLOCKING_CUDA_TRAVERSER_INSTANTIATIONS
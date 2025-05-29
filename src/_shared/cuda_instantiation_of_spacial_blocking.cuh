#define TRAVERSER_TYPE \
    cellato::traversers::cuda::spacial_blocking::traverser< \
        cellato::evaluators::standard::evaluator<AUTOMATON_NAMESPACE::config::cell_state, AUTOMATON_NAMESPACE::config::algorithm>, \
        cellato::memory::grids::standard::grid<AUTOMATON_NAMESPACE::config::cell_state, cellato::memory::grids::device::CPU>, \
        Y_TILE_SIZE, X_TILE_SIZE \
    >

template class TRAVERSER_TYPE;
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::spacial_blocking::_run_mode::QUIET>(int);
template void TRAVERSER_TYPE::run_kernel<cellato::traversers::cuda::spacial_blocking::_run_mode::VERBOSE>(int);

#undef TRAVERSER_TYPE
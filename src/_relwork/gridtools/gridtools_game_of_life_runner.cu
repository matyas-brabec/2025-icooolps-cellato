#include "gridtools_game_of_life_runner.inl"

namespace gridtools::game_of_life {

std::unique_ptr<real_runner> create_CUDA_runner() {
    return std::make_unique<game_of_life_runner>();
}

}

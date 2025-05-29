#!/bin/bash

script_dir=$(dirname "$0")

# Uncomment one of these test configurations:

# Game of Life with standard grid on CUDA
args="--automaton game-of-life \
--device CUDA --layout standard --traverser simple --evaluator standard \
--warmup_rounds 3 --rounds 10 \
--steps 100 --x_size 128 --y_size 128"

# Fire automaton with bit_array grid on CUDA
#args="--automaton fire --device CUDA --layout bit_array --steps 50 --x_size 2048 --y_size 2048"

# Wire automaton with bit_array grid and spatial blocking
#args="--automaton wire --device CUDA --layout bit_array --traverser spacial_blocking --x_tile_size 8 --y_tile_size 8 --steps 200 --x_size 4096 --y_size 4096"

# Greenberg automaton on CPU with bit_plates
#args="--automaton greenberg --device CPU --layout bit_plates --evaluator bit_plates --steps 150 --x_size 512 --y_size 512 --precision 64"

# Game of Life on CPU for comparison with CUDA
#args="--automaton game_of_life --device CPU --layout standard --steps 100 --x_size 1024 --y_size 1024"

# Visualize output with print option
#args="--automaton game_of_life --device CUDA --layout bit_array --steps 20 --x_size 32 --y_size 32 --print"

should_remove=$1

if [ "$should_remove" == "clean" ]; then
    echo "Removing old build..."
    rm -r $script_dir/../bin
fi

cd $script_dir
srun -p gpu-short -A kdss --cpus-per-task=32 --mem=64GB --time=2:00:00 --gres=gpu:L40 make run ARGS="$args"
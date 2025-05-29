#!/usr/bin/env python3
import subprocess
import os
import sys
import re

# Define automata to test
AUTOMATA = ["game-of-life", "forest-fire", "wire", "greenberg-hastings"]

GREY_COLOR = "\033[90m"
RESET_COLOR = "\033[0m"

# Set consistent parameters
class CUDA:
    GRID_SIZE = 8192  # Reasonably sized grid for performance comparison
    STEPS = 1000       # Number of steps for each run
class CPU:
    GRID_SIZE = 2048   # Reasonably sized grid for performance comparison
    STEPS = 30         # Number of steps for CPU runs

ROUNDS = 5         # Number of measurement rounds
WARMUP = 2         # Number of warmup rounds

def run_test(executable, args):
    """Run a single test and return its CSV output line"""
    cmd = [executable] + args.split()
    
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Extract the CSV line (last non-empty line of stdout)
        csv_lines = [line for line in process.stdout.strip().split('\n') if line.strip()]
        if csv_lines:
            return csv_lines[-1]  # Return the last line which is the CSV data
        else:
            print(f"No output from command: {' '.join(cmd)}", file=sys.stderr)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        return None

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, "..", "..")
    executable = os.path.join(project_dir, "bin/cellato")
    
    # Check if executable exists
    if not os.path.exists(executable):
        print(f"Error: Executable not found at {executable}", file=sys.stderr)
        print("Try running 'make' first", file=sys.stderr)
        return 1
    
    # Get the CSV header
    header = None
    try:
        result = subprocess.run(
            [executable, "--print_csv_header"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        header = result.stdout.strip()
        print(header)  # Print CSV header to stdout
    except subprocess.CalledProcessError as e:
        print(f"Error getting CSV header: {e}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        return 1
    
    # Run the tests
    for automaton in AUTOMATA:
        print(f"{GREY_COLOR}Running tests for {automaton}...{RESET_COLOR}", file=sys.stderr)
        
        # Test cases: baseline and standard on both CPU and CUDA
        test_cases_cpu = [
            # Baseline CPU
            f"--automaton {automaton} --seed 42 --device CPU --reference_impl baseline --x_size {CPU.GRID_SIZE} --y_size {CPU.GRID_SIZE} --steps {CPU.STEPS} --rounds {ROUNDS} --warmup_rounds {WARMUP}",

            # Standard CPU
            f"--automaton {automaton} --seed 42 --device CPU --traverser simple --evaluator standard --layout standard --x_size {CPU.GRID_SIZE} --y_size {CPU.GRID_SIZE} --steps {CPU.STEPS} --rounds {ROUNDS} --warmup_rounds {WARMUP}",

        ]

        test_cases_gpu = [
            (
                # Baseline CUDA
                f"--automaton {automaton} --seed 42 --device CUDA --reference_impl baseline --x_size {CUDA.GRID_SIZE} --y_size {CUDA.GRID_SIZE} --steps {CUDA.STEPS} --rounds {ROUNDS} --warmup_rounds {WARMUP} --cuda_block_size_y {block_y}",

                # Standard CUDA
                f"--automaton {automaton} --seed 42 --device CUDA --traverser simple --evaluator standard --layout standard --x_size {CUDA.GRID_SIZE} --y_size {CUDA.GRID_SIZE} --steps {CUDA.STEPS} --rounds {ROUNDS} --warmup_rounds {WARMUP} --cuda_block_size_y {block_y}"
            ) 
            # for block_y in [1, 2, 4, 8, 16, 32]
            for block_y in [4]  # has been shown to be the best for all automata
        ]

        #flatten the list of test cases
        test_cases = test_cases_cpu + [case for sublist in test_cases_gpu for case in sublist]
        
        for test_case in test_cases:
            print(f"{GREY_COLOR}Running: {test_case}{RESET_COLOR}", file=sys.stderr)
            csv_line = run_test(executable, test_case)
            if csv_line:
                print(csv_line)  # Print CSV data to stdout
            else:
                print(f"Failed to get results for: {test_case}", file=sys.stderr)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

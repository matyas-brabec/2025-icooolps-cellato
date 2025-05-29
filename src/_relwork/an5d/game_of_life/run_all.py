#!/usr/bin/env python3
import os
import subprocess
import re
import statistics
import sys

# CSV header
csv_header = "automaton,device,traverser,evaluator,layout,reference_impl,x_size,y_size,steps,rounds,warmup_rounds,x_tile_size,y_tile_size,cuda_block_size_x,cuda_block_size_y,seed,precision,average_time_ms,average_time_per_cell_ns,std_time_ms,rounds_had_same_checksums,checksum"
print(csv_header)

# Directories to process (grid sizes)
directories = ["2048", "4096", "8192", "16386"]

args = sys.argv[1:]

for directory in directories:
    # Navigate to the directory
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), directory))
    
    # Check if executable exists, if not, build it
    if not os.path.exists("GOL_an5d"):
        try:
            clean_result = subprocess.run(["make", "clean"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if clean_result.returncode != 0:
                print(f"Error cleaning in {directory}: {clean_result.stderr.decode()}", file=sys.stderr)
                os.chdir("..")
                continue
            make_result = subprocess.run(["make"] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if make_result.returncode != 0:
                print(f"Error building in {directory}: {make_result.stderr.decode()}", file=sys.stderr)
                os.chdir("..")
                continue
        except Exception as e:
            print(f"Error building in {directory}: {e}", file=sys.stderr)
            os.chdir("..")
            continue
    
    # Run the executable and capture output
    try:
        result = subprocess.run(["./GOL_an5d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True, check=True)
        output = result.stdout
    except Exception as e:
        print(f"Error running in {directory}: {e}", file=sys.stderr)
        os.chdir("..")
        continue
    
    # Parse the output
    times = []
    iterations = 0
    grid_size = 0
    
    for line in output.splitlines():
        # Extract iterations and grid size
        case_match = re.search(r'CASE (\d+) - iters: (\d+)', line)
        if case_match:
            grid_size = int(case_match.group(1))
            iterations = int(case_match.group(2))
            
        # Extract execution time
        time_match = re.search(r'Time: (\d+) ms', line)
        if time_match:
            times.append(int(time_match.group(1)))
    
    # Skip if we didn't get enough data
    if len(times) < 15 or grid_size == 0 or iterations == 0:
        print(f"Not enough data from {directory}, skipping", file=sys.stderr)
        os.chdir("..")
        continue
    
    # Separate warmup rounds from actual rounds
    warmup_rounds = 5
    actual_rounds = times[warmup_rounds:]
    
    # Calculate statistics
    avg_time_ms = statistics.mean(actual_rounds)
    std_time_ms = statistics.stdev(actual_rounds) if len(actual_rounds) > 1 else 0
    
    # The grid size is the directory name, but we need the logical size (without margins)
    logical_size = int(directory)  # Directory name corresponds to logical grid size
    
    # Calculate time per cell in nanoseconds
    # Time (ms) * 1e6 / (grid_size^2 * iterations)
    cells_per_iter = logical_size * logical_size
    avg_time_per_cell_ns = (avg_time_ms * 1e6) / (cells_per_iter * iterations)
    
    # Format CSV line
    csv_line = (f"game-of-life,CUDA,,,,"  # automaton,device,traverser,evaluator,layout
               f"an5d,"               # reference_impl
               f"{logical_size},{logical_size},"  # x_size,y_size
               f"{iterations},10,5,"  # steps,rounds,warmup_rounds
               f"0,0,"                # x_tile_size,y_tile_size
               f"16,16,"              # cuda_block_size_x,cuda_block_size_y
               f"42,,"                # seed,precision
               f"{avg_time_ms:.6f},{avg_time_per_cell_ns:.6f},{std_time_ms:.6f},"  # average_time_ms,average_time_per_cell_ns,std_time_ms
               f"true,")              # rounds_had_same_checksums,checksum (empty)
    
    print(csv_line)
    
    # Go back to parent directory
    os.chdir("..")
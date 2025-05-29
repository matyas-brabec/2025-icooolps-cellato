#!/usr/bin/env python3
import subprocess
import time
import os
import sys
import re
from datetime import datetime
import csv
import shutil

# Define automata to test
AUTOMATA = ["game-of-life", "forest-fire", "wire", "greenberg-hastings"]

smallest_dim = 6720

class Tester:
    def __init__(self):
        self.automata = AUTOMATA
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.join(self.base_dir, "..", "..")
        self.executable = os.path.join(self.project_dir, "bin/cellib_experiments")
        
        # Test configurations
        self.test_configs = {
            "check": {
                "rounds": 2,
                "warmup_rounds": 0,
                "grid_sizes": [(smallest_dim, smallest_dim)],
                "seed": 42,
                "steps": {
                    # Same steps for verification
                    "game-of-life": 16,
                    "forest-fire": 16,
                    "wire": 16,
                    "greenberg-hastings": 16
                }
            },
            "measurement": {
                "rounds": 10,
                "warmup_rounds": 5,
                "grid_sizes": [(smallest_dim, smallest_dim), (smallest_dim * 2, smallest_dim * 2), (smallest_dim * 3, smallest_dim * 3)],
                "seed": 42,
                "steps": {
                    # CPU implementations (slower)
                    "CPU": {
                        "game-of-life": 20,
                        "forest-fire": 10,
                        "wire": 30,
                        "greenberg-hastings": 5
                    },
                    # CUDA implementations (faster)
                    "CUDA": {
                        "game-of-life": 200,
                        "forest-fire": 100,
                        "wire": 300,
                        "greenberg-hastings": 50
                    }
                }
            }
        }
    
    def cases(self, automaton):
        """Define all test cases for a given automaton"""
        return [
            # Baseline test (no traverser, no evaluator, no layout)
            f"--automaton {automaton} --device CPU --reference_impl baseline --x_size 512 --y_size 512 --steps 100 --precision 32 --seed 42",
            f"--automaton {automaton} --device CUDA --reference_impl baseline --x_size 512 --y_size 512 --steps 100 --precision 32 --seed 42",

            # CPU tests
            f"--automaton {automaton} --device CPU --traverser simple --evaluator standard --layout standard --x_size 512 --y_size 512 --steps 100 --seed 42",
            f"--automaton {automaton} --device CPU --traverser simple --evaluator bit_array --layout bit_array --x_size 512 --y_size 512 --steps 100 --precision 32 --seed 42",
            f"--automaton {automaton} --device CPU --traverser simple --evaluator bit_plates --layout bit_plates --x_size 512 --y_size 512 --steps 100 --precision 32 --seed 42",
            
            # CUDA standard tests
            f"--automaton {automaton} --device CUDA --traverser simple --evaluator standard --layout standard --x_size 512 --y_size 512 --steps 100 --seed 42",
            
            # CUDA with spatial blocking tests (different tile sizes)
            f"--automaton {automaton} --device CUDA --traverser spacial_blocking --evaluator standard --layout standard --x_size 512 --y_size 512 --steps 100 --x_tile_size 1 --y_tile_size 1 --seed 42",
            f"--automaton {automaton} --device CUDA --traverser spacial_blocking --evaluator standard --layout standard --x_size 512 --y_size 512 --steps 100 --x_tile_size 1 --y_tile_size 2 --seed 42",
            f"--automaton {automaton} --device CUDA --traverser spacial_blocking --evaluator standard --layout standard --x_size 512 --y_size 512 --steps 100 --x_tile_size 1 --y_tile_size 4 --seed 42",
            
            # CUDA bit array tests with different precision
            f"--automaton {automaton} --device CUDA --traverser simple --evaluator bit_array --layout bit_array --x_size 512 --y_size 512 --steps 100 --precision 32 --seed 42",
            f"--automaton {automaton} --device CUDA --traverser simple --evaluator bit_array --layout bit_array --x_size 512 --y_size 512 --steps 100 --precision 64 --seed 42",
            
            # CUDA bit plates tests with different precision
            f"--automaton {automaton} --device CUDA --traverser simple --evaluator bit_plates --layout bit_plates --x_size 512 --y_size 512 --steps 100 --precision 32 --seed 42",
            f"--automaton {automaton} --device CUDA --traverser simple --evaluator bit_plates --layout bit_plates --x_size 512 --y_size 512 --steps 100 --precision 64 --seed 42",
        ]
    
    def compile_once(self):
        """Compile the program once at the beginning"""
        print("Compiling the program...")
        try:
            # Clean and rebuild
            subprocess.run(["make", "clean"], cwd=self.project_dir + "/src", check=True)
            subprocess.run(["make", "-j"], cwd=self.project_dir + "/src", check=True)
            print("✅ Compilation successful")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Compilation failed: {e}")
            return False
    
    def get_csv_header(self):
        """Get the CSV header by executing the program with --print_csv_header"""
        try:
            result = subprocess.run(
                [self.executable, "--print_csv_header"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error getting CSV header: {e}")
            print(f"STDERR: {e.stderr}")
            return None
    
    def run_case(self, args, report_file):
        """Run a single test case and save results to the report file"""
        # Extract case info for logging

        try:
            reference = re.search(r'--reference_impl (\w+)', args).group(1)
        except AttributeError:
            reference = "none"

        automaton = re.search(r'--automaton (\w+)', args).group(1)
        device = re.search(r'--device (\w+)', args).group(1)

        if reference == "none":
            layout = re.search(r'--layout (\w+)', args).group(1)
            evaluator = re.search(r'--evaluator (\w+)', args).group(1)
            traverser = re.search(r'--traverser (\w+)', args).group(1)
         
        x_size = re.search(r'--x_size (\d+)', args).group(1)
        y_size = re.search(r'--y_size (\d+)', args).group(1)
        steps = re.search(r'--steps (\d+)', args).group(1)
        
        if reference == "none":
            case_desc = f"{automaton} {device}/{traverser}/{evaluator}/{layout} {x_size}x{y_size} steps={steps}"
        else:
            case_desc = f"{automaton} reference={reference} {device} {x_size}x{y_size} steps={steps}"
        print(f"\nRunning case: {case_desc}")
        
        # Execute the program (without rebuilding)
        cmd = [self.executable] + args.split()
        print(f"Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"❌ Error running case: {case_desc}")
                print(f"STDERR: {stderr}")
                return None, None
            
            # Extract the CSV line (last non-empty line of stdout)
            csv_lines = [line for line in stdout.strip().split('\n') if line.strip()]
            if csv_lines:
                csv_line = csv_lines[-1]
                # Extract checksum from CSV line
                checksum_match = re.search(r',([^,]+)$', csv_line)
                if checksum_match:
                    checksum = checksum_match.group(1)
                
                # Write CSV line to report file
                with open(report_file, 'a') as f:
                    f.write(csv_line + '\n')
                
                print(f"✅ Case completed: {case_desc}")
                return csv_line, checksum
            else:
                print(f"❌ No CSV output from case: {case_desc}")
                return None, None
                
        except Exception as e:
            print(f"❌ Exception running case: {case_desc}")
            print(f"Error: {str(e)}")
            return None, None
    
    def run_check_test(self):
        """Run tests to verify checksums are consistent across implementations"""
        print("\n=== RUNNING CHECKSUM VERIFICATION TEST ===")
        
        # Create timestamp for the report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.project_dir, 'results', f"report-check-{timestamp}.csv")
        
        # Get CSV header and write to report file
        header = self.get_csv_header()
        if header:
            with open(report_file, 'w') as f:
                f.write(header + '\n')
        
        config = self.test_configs["check"]
        x_size, y_size = config["grid_sizes"][0]
        
        # Track checksums by automaton to verify consistency
        automaton_checksums = {automaton: {} for automaton in self.automata}
        
        for automaton in self.automata:
            print(f"\n--- Testing automaton: {automaton} ---")
            steps = config["steps"][automaton]
            
            for base_case in self.cases(automaton):
                # Update case with check test parameters
                case = base_case
                case = re.sub(r'--x_size \d+', f'--x_size {x_size}', case)
                case = re.sub(r'--y_size \d+', f'--y_size {y_size}', case)
                case = re.sub(r'--steps \d+', f'--steps {steps}', case)
                case += f" --rounds {config['rounds']} --warmup_rounds {config['warmup_rounds']}"
                
                # Run the case
                _, checksum = self.run_case(case, report_file)
                
                if checksum:
                    # Store result for this implementation
                    try:
                        impl_key = re.search(r'--device (\w+) --traverser (\w+) --evaluator (\w+) --layout (\w+)', case).group(0)
                    except AttributeError:
                        impl_key = re.search(r'--device (\w+) --reference_impl (\w+)', case).group(0)

                    automaton_checksums[automaton][impl_key] = checksum
        
        # Check if all implementations produce the same checksum for each automaton
        print("\n=== CHECKSUM VERIFICATION RESULTS ===")
        for automaton, checksums in automaton_checksums.items():
            if len(checksums) == 0:
                print(f"❌ {automaton}: No checksums collected")
                continue
                
            unique_checksums = set(checksums.values())
            if len(unique_checksums) == 1:
                print(f"✅ {automaton}: All {len(checksums)} implementations produced the same checksum: {next(iter(unique_checksums))}")
            else:
                print(f"❌ {automaton}: Found {len(unique_checksums)} different checksums across {len(checksums)} implementations:")
                for impl, cksum in checksums.items():
                    print(f"   - {impl}: {cksum}")
        
        print(f"\nReport saved to: {report_file}")
        return report_file
    
    def run_measurement_test(self):
        """Run performance measurement tests with different grid sizes"""
        print("\n=== RUNNING PERFORMANCE MEASUREMENT TEST ===")
        
        # Create timestamp for the report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.project_dir, 'results', f"report-measurement-{timestamp}.csv")
        
        # Get CSV header and write to report file
        header = self.get_csv_header()
        if header:
            with open(report_file, 'w') as f:
                f.write(header + '\n')
        
        config = self.test_configs["measurement"]
        
        for automaton in self.automata:
            print(f"\n--- Testing automaton: {automaton} ---")
            
            for base_case in self.cases(automaton):
                # Extract device to determine step count
                device = re.search(r'--device (\w+)', base_case).group(1)
                steps = config["steps"][device][automaton]
                
                # For each grid size
                for x_size, y_size in config["grid_sizes"]:
                    # Skip large grid sizes for CPU tests (they take too long)
                    if device == "CPU" and x_size > 2048:
                        print(f"Skipping large grid {x_size}x{y_size} for CPU test")
                        continue
                    
                    # Update case with measurement test parameters
                    case = base_case
                    case = re.sub(r'--x_size \d+', f'--x_size {x_size}', case)
                    case = re.sub(r'--y_size \d+', f'--y_size {y_size}', case)
                    case = re.sub(r'--steps \d+', f'--steps {steps}', case)
                    case += f" --rounds {config['rounds']} --warmup_rounds {config['warmup_rounds']}"
                    
                    # Run the case
                    self.run_case(case, report_file)
        
        print(f"\nReport saved to: {report_file}")
        return report_file
    
    def run(self):
        """Run all test types"""
        print("Starting cellular automaton tests...")
        
        start_time = time.time()
        
        # Compile once at the beginning
        if not self.compile_once():
            print("❌ Compilation failed. Exiting tests.")
            return
        
        # Run both test types
        check_report = self.run_check_test()
        measurement_report = self.run_measurement_test()
        
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n=== TESTING COMPLETE ===")
        print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Check report: {check_report}")
        print(f"Measurement report: {measurement_report}")

if __name__ == "__main__":
    tester = Tester()
    tester.run()

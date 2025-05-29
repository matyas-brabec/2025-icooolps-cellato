import sys
import pandas as pd
import re

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

def load_data(csv_file):
    """Load and parse the CSV file."""
    return pd.read_csv(csv_file)

def process_data(df):
    """Process the dataframe to extract the required information."""
    # Create a dictionary to store the processed data
    result = {}
    
    # Group by grid size
    grid_sizes = sorted(set(zip(df['x_size'], df['y_size'])))
    
    for x_size, y_size in grid_sizes:
        grid_df = df[(df['x_size'] == x_size) & (df['y_size'] == y_size)]
        
        # Create entries for each automaton
        grid_result = {}
        for automaton in sorted(grid_df['automaton'].unique()):
            automaton_df = grid_df[grid_df['automaton'] == automaton]
            
            # Get the true baseline (reference_impl = "baseline")
            true_baseline = automaton_df[automaton_df['reference_impl'] == 'baseline']
            if not true_baseline.empty:
                true_baseline_ns = true_baseline.iloc[0]['average_time_per_cell_ns']
            else:
                true_baseline_ns = None
            
            # Get standard implementation (CUDA + simple + standard evaluator + standard layout)
            standard_impl = automaton_df[(automaton_df['device'] == 'CUDA') & 
                                        (automaton_df['traverser'] == 'simple') &
                                        (automaton_df['evaluator'] == 'standard') & 
                                        (automaton_df['layout'] == 'standard') &
                                        (automaton_df['reference_impl'] != 'baseline')]
            if not standard_impl.empty:
                standard_impl_ns = standard_impl.iloc[0]['average_time_per_cell_ns']
            else:
                standard_impl_ns = None
            
            # Get bit_array and bit_plates results
            bit_array_32 = automaton_df[(automaton_df['evaluator'] == 'bit_array') & 
                                       (automaton_df['layout'] == 'bit_array') & 
                                       (automaton_df['precision'] == 32)]
            bit_array_64 = automaton_df[(automaton_df['evaluator'] == 'bit_array') & 
                                       (automaton_df['layout'] == 'bit_array') & 
                                       (automaton_df['precision'] == 64)]
            bit_plates_32 = automaton_df[(automaton_df['evaluator'] == 'bit_plates') & 
                                        (automaton_df['layout'] == 'bit_plates') & 
                                        (automaton_df['precision'] == 32)]
            bit_plates_64 = automaton_df[(automaton_df['evaluator'] == 'bit_plates') & 
                                        (automaton_df['layout'] == 'bit_plates') & 
                                        (automaton_df['precision'] == 64)]
            
            # Extract the values
            results = {
                'true_baseline': true_baseline_ns,
                'standard': standard_impl_ns,
                'bit_array_32': None if bit_array_32.empty else bit_array_32.iloc[0]['average_time_per_cell_ns'],
                'bit_array_64': None if bit_array_64.empty else bit_array_64.iloc[0]['average_time_per_cell_ns'],
                'bit_plates_32': None if bit_plates_32.empty else bit_plates_32.iloc[0]['average_time_per_cell_ns'],
                'bit_plates_64': None if bit_plates_64.empty else bit_plates_64.iloc[0]['average_time_per_cell_ns']
            }
            
            grid_result[automaton] = results
        
        result[(x_size, y_size)] = grid_result
    
    return result

def visible_len(s):
    """Calculate the visible length of a string by removing ANSI color codes."""
    # Remove all ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', s))

def format_cell(value, baseline=None):
    """Format a cell value, including speedup if baseline is available."""
    if value is None:
        return "N/A"
    
    if baseline is None:
        return f"{value:.6f} ns"
    
    speedup = baseline / value
    color = Colors.GREEN if speedup > 1 else Colors.RED
    return f"{value:.6f} ns ({color}{speedup:.2f}x{Colors.RESET})"

def display_results(data):
    """Display the results in a formatted table."""
    automaton_display = {
        'game-of-life': 'Game of Life',
        'forest-fire': 'Forest Fire',
        'wire': 'Wire',
        'greenberg-hastings': 'Greenberg'
    }
    
    for (x_size, y_size), grid_data in sorted(data.items()):
        print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 100}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}Grid Size: {x_size} x {y_size}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 100}{Colors.RESET}")
        
        # Column widths
        col_widths = [15, 25, 25, 25, 25, 25, 25]
        columns = ["Automaton", "True Baseline", "Standard", "Bit Array (32-bit)", "Bit Array (64-bit)", "Bit Plates (32-bit)", "Bit Plates (64-bit)"]
        
        # Print header with yellow color
        header = f"{Colors.YELLOW}{Colors.BOLD}{columns[0]:{col_widths[0]}}"
        for i, col in enumerate(columns[1:], 1):
            header += f"{col:{col_widths[i]}}"
        header += Colors.RESET
        print(header)
        
        print(f"{Colors.YELLOW}{'-' * 140}{Colors.RESET}")
        
        # Print each row
        for automaton, results in sorted(grid_data.items()):
            true_baseline = results['true_baseline']
            display_name = automaton_display.get(automaton, automaton)
            
            # Prepare cell values with colors
            automaton_cell = f"{Colors.BOLD}{display_name}{Colors.RESET}"
            true_baseline_cell = format_cell(true_baseline)
            
            # Standard implementation compared to true baseline
            standard_cell = format_cell(results['standard'], true_baseline)
            
            # Other implementations compared to true baseline
            bit_array_32_cell = format_cell(results['bit_array_32'], true_baseline)
            bit_array_64_cell = format_cell(results['bit_array_64'], true_baseline)
            bit_plates_32_cell = format_cell(results['bit_plates_32'], true_baseline)
            bit_plates_64_cell = format_cell(results['bit_plates_64'], true_baseline)
            
            # Calculate padding for each cell based on visible length
            auto_visible_len = visible_len(automaton_cell)
            auto_padding = ' ' * (col_widths[0] - auto_visible_len)
            
            baseline_visible_len = visible_len(true_baseline_cell)
            baseline_padding = ' ' * (col_widths[1] - baseline_visible_len)
            
            standard_visible_len = visible_len(standard_cell)
            standard_padding = ' ' * (col_widths[2] - standard_visible_len)
            
            bit_array_32_visible_len = visible_len(bit_array_32_cell)
            bit_array_32_padding = ' ' * (col_widths[3] - bit_array_32_visible_len)
            
            bit_array_64_visible_len = visible_len(bit_array_64_cell)
            bit_array_64_padding = ' ' * (col_widths[4] - bit_array_64_visible_len)
            
            bit_plates_32_visible_len = visible_len(bit_plates_32_cell)
            bit_plates_32_padding = ' ' * (col_widths[5] - bit_plates_32_visible_len)
            
            bit_plates_64_visible_len = visible_len(bit_plates_64_cell)
            bit_plates_64_padding = ' ' * (col_widths[6] - bit_plates_64_visible_len)
            
            # Build row with correct spacing
            row = f"{automaton_cell}{auto_padding}{true_baseline_cell}{baseline_padding}{standard_cell}{standard_padding}{bit_array_32_cell}{bit_array_32_padding}{bit_array_64_cell}{bit_array_64_padding}{bit_plates_32_cell}{bit_plates_32_padding}{bit_plates_64_cell}{bit_plates_64_padding}"
            print(row)
        
        print()

def main():
    if len(sys.argv) < 2:
        print(f"{Colors.RED}Usage: python compare.py <csv_file>{Colors.RESET}")
        return
    
    csv_file = sys.argv[1]
    df = load_data(csv_file)
    data = process_data(df)
    display_results(data)

if __name__ == "__main__":
    main()

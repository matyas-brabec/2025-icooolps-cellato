# Cellular Beauty Performance Analysis Scripts

This directory contains scripts for analyzing and visualizing the performance of cellular automata implementations.

## Environment Setup

### Prerequisites

- Python 3.8 or newer
- pip (Python package installer)
- virtualenv (recommended)

### Setting up the Python Environment

1. **Create a virtual environment** (recommended)

   ```bash
   # Navigate to the scripts directory
   cd <repo>/src/_scripts
   
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   source venv/bin/activate  # On Linux/macOS
   # or
   venv\Scripts\activate  # On Windows
   ```

2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Analysis Scripts

### Baseline vs Standard Comparison

This script compares the performance of baseline and standard implementations:

```bash
python plot_baseline_vs_standard.py <path_to_csv_file>
```

Example:
```bash
# First run the benchmark to generate CSV data
python baseline_vs_standard.py > ../../results/baseline_vs_standard_data.csv

# Then create the visualization
python plot_baseline_vs_standard.py ../../results/baseline_vs_standard_data.csv
```

The plot will be saved to `../../results/baseline_vs_standard_comparison.png`

### Common Issues

- **File not found error**: Make sure you're providing the correct path to an existing CSV file.
- **CSV format error**: Ensure your CSV has the expected columns including automaton, device, implementation, etc.
- **Missing dependencies**: If you encounter import errors, check that all required packages are installed with `pip list`.

## CSV Format

The expected CSV format for the plotting scripts includes these columns:


#!/usr/bin/env python3
# Written by Claude.ai

import argparse
import os
import subprocess
import tempfile
import sys
from typing import List, Tuple, Dict, Literal
import re

# Define a type for test status
TestStatus = Literal["PASS", "FAIL", "SIM_ERROR"]

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Regular expression to strip ANSI color codes
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run hardware simulations and compare outputs')
    parser.add_argument('-t', '--test_dir', default='dv/test_vectors',
                        help='Directory containing test vectors (default: dv/test_vectors)')
    parser.add_argument('-s', '--simulator', default='./obj_dir/Vtiny_nn_top_tb',
                        help='Path to simulator binary (default: ./obj_dir/Vtiny_nn_top_tb)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    return parser.parse_args()

def find_test_files(test_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find all test input files and their corresponding expected output files.

    Returns a list of tuples (test_name, input_file, expected_output_file)
    """
    test_files = []

    # Ensure the test directory exists
    if not os.path.isdir(test_dir):
        print(f"Error: Test directory '{test_dir}' not found.")
        sys.exit(1)

    # Find all input files matching the pattern test_*.hex
    for file in os.listdir(test_dir):
        if file.startswith('test_') and file.endswith('.hex') and not file.endswith('_expected.hex'):
            input_file = os.path.join(test_dir, file)

            # Extract test name - remove 'test_' prefix and '.hex' suffix
            test_name = file[5:-4]  # Extract name between 'test_' and '.hex'

            # Construct expected output filename
            expected_file = os.path.join(test_dir, f"test_{test_name}_expected.hex")

            # Check if expected output file exists
            if os.path.isfile(expected_file):
                test_files.append((test_name, input_file, expected_file))
            else:
                print(f"Warning: Expected output file '{expected_file}' not found for input '{input_file}'.")

    return test_files

def run_simulation(simulator_path: str, input_file: str, verbose: bool) -> Tuple[bool, str]:
    """
    Run the hardware simulation with the given input file.

    Returns a tuple of (success, output_file_path).
    If simulation fails, success will be False and output_file_path will be None.
    """
    # Create a temporary file for the output
    with tempfile.NamedTemporaryFile(delete=False, suffix='.hex') as temp_file:
        output_file = temp_file.name

    # Build the command
    cmd = [
        simulator_path,
        f"+test_data={input_file}",
        f"+out={output_file}"
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    # Run the simulation
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (True, output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error: Simulation failed with exit code {e.returncode}")
        if os.path.exists(output_file):
            os.unlink(output_file)
        return (False, None)

def compare_outputs(sim_output: str, expected_output: str, verbose: bool) -> bool:
    """
    Compare the simulation output with the expected output.

    Returns True if the outputs match, False otherwise.
    """
    cmd = [
        'dv/out_compare.py',
        '-s', sim_output,
        '-e', expected_output
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    # Run the comparison
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from a string."""
    return ANSI_ESCAPE.sub('', text)

def colorize(text: str, color: str, use_color: bool = True) -> str:
    """Apply color to text if use_color is True."""
    if use_color:
        return f"{color}{text}{Colors.RESET}"
    return text

def print_results_table(results: Dict[str, TestStatus], use_color: bool = True):
    """Print a formatted ASCII table of the test results."""
    # Find the longest test name for formatting
    max_name_len = max(len(name) for name in results.keys())
    result_col_width = 10  # Width for the result column

    # Create the table structure
    separator = f"+{'-' * (max_name_len + 2)}+{'-' * (result_col_width + 2)}+"

    # Print the table header
    header_test = "Test Name"
    header_result = "Result"

    if use_color:
        header_test = colorize(header_test, Colors.BOLD)
        header_result = colorize(header_result, Colors.BOLD)

    print(separator)
    print(f"| {header_test:<{max_name_len+8}} | {header_result:<{result_col_width+8}} |")
    print(separator)

    # Print each result
    pass_count = 0
    sim_error_count = 0

    for name, status in results.items():
        result_text = status
        if status == "PASS":
            pass_count += 1
            result_styled = colorize(result_text, Colors.GREEN, use_color)
        elif status == "SIM_ERROR":
            sim_error_count += 1
            result_styled = colorize(result_text, Colors.YELLOW, use_color)
        else:  # FAIL
            result_styled = colorize(result_text, Colors.RED, use_color)

        print(f"| {name:<{max_name_len}} | {result_styled}{' ' * (result_col_width - len(result_text))} |")

    # Print the footer
    print(separator)

    # Print summary
    total = len(results)
    fail_count = total - pass_count - sim_error_count

    pass_percentage = pass_count/total*100 if total > 0 else 0

    summary_text = f"Summary: {pass_count}/{total} tests passed ({pass_percentage:.1f}%)"
    print(colorize(summary_text, Colors.GREEN if pass_count == total else Colors.RESET, use_color))

    if sim_error_count > 0:
        sim_error_text = f"         {sim_error_count}/{total} simulation errors"
        print(colorize(sim_error_text, Colors.YELLOW, use_color))

    if fail_count > 0:
        fail_text = f"         {fail_count}/{total} comparison failures"
        print(colorize(fail_text, Colors.RED, use_color))

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()

    # Determine if we should use colors
    use_color = not args.no_color and sys.stdout.isatty()

    # Find test files
    test_files = find_test_files(args.test_dir)

    if not test_files:
        print("No test files found.")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(test_files)} test files.")

    # Run simulations and compare outputs
    results = {}

    for test_name, input_file, expected_file in test_files:
        if args.verbose:
            print(f"\nProcessing test: {test_name}")

        # Run simulation
        sim_success, output_file = run_simulation(args.simulator, input_file, args.verbose)

        if not sim_success:
            results[test_name] = "SIM_ERROR"
            continue

        # Compare outputs
        comparison_passed = compare_outputs(output_file, expected_file, args.verbose)
        results[test_name] = "PASS" if comparison_passed else "FAIL"

        # Clean up temporary output file
        os.unlink(output_file)

    # Print results
    print_results_table(results, use_color)

if __name__ == "__main__":
    main()
